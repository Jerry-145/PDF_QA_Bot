from fastapi import FastAPI, Request, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from groq import Groq
from dotenv import load_dotenv
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
from slowapi import Limiter
from slowapi.util import get_remote_address
import os 
import re
import uvicorn
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from uuid import uuid4
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Authentication imports
from database import engine, get_db
from auth.models import Base, User
from auth.router import router as auth_router
from auth.middleware import (
    get_current_user, 
    get_optional_user,
    require_upload_permission,
    require_ask_permission,
    require_summarize_permission,
    require_compare_permission,
    require_view_documents_permission
)

# ===============================
# APP SETUP
# ===============================
load_dotenv()

# ===============================
# APP INITIALIZATION 
# ===============================
app = FastAPI(
    title="PDF QA Bot API",
    description="Secure PDF Question-Answering Bot with Authentication",
    version="2.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# DATABASE INITIALIZATION
# ===============================
# Create database tables
Base.metadata.create_all(bind=engine)

# Include authentication router
app.include_router(auth_router)

# ===============================
# HEALTH AND READINESS ENDPOINTS
# ===============================
@app.get("/healthz")
def health_check():
    """Health check endpoint - returns 200 if service is running"""
    return {"status": "healthy", "service": "pdf-qa-rag"}

@app.get("/readyz")
def readiness_check():
    """Readiness check endpoint - returns 200 if service is ready to handle requests"""
    from fastapi import HTTPException
    
    try:
        # Check if embedding model is available
        if embedding_model is None:
            raise HTTPException(status_code=503, detail={"status": "not ready", "error": "Embedding model not initialized"})
        
        # Check if we can load generation models
        try:
            load_generation_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail={"status": "not ready", "error": f"Generation model not ready: {str(e)}"})
        
        # All checks passed
        return {
            "status": "ready", 
            "service": "pdf-qa-rag",
            "components": {
                "embedding_model": "ready",
                "generation_model": "ready",
                "vector_store": "available" if VECTOR_STORE is not None else "empty"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail={"status": "not ready", "error": str(e)})

# ===============================
# RATE LIMITING SETUP 
# ===============================
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ===============================
# GLOBAL STATE (Multi-document support from remote)
# ===============================
HF_GENERATION_MODEL = os.getenv("HF_GENERATION_MODEL", "google/flan-t5-small")
LLM_GENERATION_TIMEOUT = int(os.getenv("LLM_GENERATION_TIMEOUT", "30"))
SESSION_TIMEOUT = 3600

# ---------------------------------------------------------------------------
# GLOBAL STATE MANAGEMENT (Thread-safe, Multi-user support)
# ---------------------------------------------------------------------------
# Per-user/session storage with proper cleanup and locking
sessions = {}  # {session_id: {"vectorstore": FAISS, "upload_time": datetime}}
sessions_lock = threading.RLock()  # Thread-safe access to sessions

# Load local embedding model (unchanged — FAISS retrieval stays the same)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------------------------------------------
# SESSION MANAGEMENT UTILITIES (Thread-safe, Multi-user support)
# ---------------------------------------------------------------------------

def get_session_vectorstore(session_id: str):
    """
    Safely retrieves vectorstore for a session.
    Returns (vectorstore, upload_time) or (None, None) if not found.
    """
    with sessions_lock:
        if session_id in sessions:
            session_data = sessions[session_id]
            return session_data.get("vectorstore"), session_data.get("upload_time")
        return None, None


def set_session_vectorstore(session_id: str, vectorstore, upload_time: str):
    """
    Safely stores vectorstore for a session.
    Clears old session if it exists (replaces it).
    """
    with sessions_lock:
        # Clear old session to prevent memory leaks
        if session_id in sessions:
            old_vectorstore = sessions[session_id].get("vectorstore")
            if old_vectorstore is not None:
                del old_vectorstore  # Allow garbage collection
        
        # Store new session
        sessions[session_id] = {
            "vectorstore": vectorstore,
            "upload_time": upload_time
        }


def clear_session(session_id: str):
    """
    Safely clears a specific session's vectorstore and data.
    """
    with sessions_lock:
        if session_id in sessions:
            old_vectorstore = sessions[session_id].get("vectorstore")
            if old_vectorstore is not None:
                del old_vectorstore  # Allow garbage collection
            del sessions[session_id]


def normalize_spaced_text(text: str) -> str:
    pattern = r"\b(?:[A-Za-z] ){2,}[A-Za-z]\b"
    return re.sub(pattern, lambda m: m.group(0).replace(" ", ""), text)


def normalize_answer(text: str) -> str:
    """
    Post-processes the LLM-generated answer.
    """
    text = normalize_spaced_text(text)
    text = re.sub(r"^(Answer[^:]*:|Context:|Question:)\s*", "", text, flags=re.I)
    return text.strip()


# ===============================
# DOCUMENT LOADERS
# ===============================
def load_pdf(file_path: str):
    return PyPDFLoader(file_path).load()


def load_txt(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return [Document(page_content=f.read())]


def load_docx(file_path: str):
    doc = docx.Document(file_path)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return [Document(page_content=text)]


def load_document(file_path: str):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext in [".txt", ".md"]:
        return load_txt(file_path)
    else:
        raise ValueError("Unsupported file format")


# ===============================
# MODEL LOADING
# ===============================
def load_generation_model():
    global generation_model, generation_tokenizer, generation_is_encoder_decoder

    if generation_model:
        return generation_tokenizer, generation_model, generation_is_encoder_decoder

    config = AutoConfig.from_pretrained(HF_GENERATION_MODEL)
    generation_is_encoder_decoder = bool(config.is_encoder_decoder)

    generation_tokenizer = AutoTokenizer.from_pretrained(HF_GENERATION_MODEL)

    if generation_is_encoder_decoder:
        generation_model = AutoModelForSeq2SeqLM.from_pretrained(HF_GENERATION_MODEL)
    else:
        generation_model = AutoModelForCausalLM.from_pretrained(HF_GENERATION_MODEL)

    if torch.cuda.is_available():
        generation_model = generation_model.to("cuda")

    generation_model.eval()
    return generation_tokenizer, generation_model, generation_is_encoder_decoder


def generate_response(prompt: str, max_new_tokens: int):
    tokenizer, model, is_enc = load_generation_model()
    device = next(model.parameters()).device

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    encoded = {k: v.to(device) for k, v in encoded.items()}

    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    if is_enc:
        return tokenizer.decode(output[0], skip_special_tokens=True)

    return tokenizer.decode(
        output[0][encoded["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


# ===============================
# REQUEST MODELS
# ===============================
class DocumentPath(BaseModel):
    filePath: str
    session_id: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str
    history: list = []

# ===============================
# UPLOAD ENDPOINT (Expected by frontend)
# ===============================
@app.post("/upload") 
@limiter.limit("10/15 minutes")
async def upload_file(
    request: Request, 
    file: UploadFile = File(...),
    current_user: User = Depends(require_upload_permission)
):
    """Upload and process PDF file (Authentication required)"""
    if not file.filename.lower().endswith('.pdf'):
        return {"error": "Only PDF files are supported"}
    
    # Save uploaded file temporarily
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{uuid4().hex}_{file.filename}")
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the PDF
        result = process_pdf_internal(file_path)
        
        # Add user info to the result for audit trail
        if result and "doc_id" in result:
            result["uploaded_by"] = current_user.username
            result["user_id"] = current_user.id
        
        # Clean up temporary file (optional - you may want to keep it)
        # os.remove(file_path)
        
        return result
        
    except Exception as e:
        return {"error": f"Upload failed: {str(e)}"}

# Optional: Legacy endpoint for backward compatibility (will be deprecated)
@app.post("/upload/anonymous", deprecated=True) 
@limiter.limit("5/15 minutes")  # More restrictive rate limit for anonymous users
async def upload_file_anonymous(
    request: Request, 
    file: UploadFile = File(...)
):
    """Upload and process PDF file (Anonymous - Deprecated)"""
    import warnings
    warnings.warn("Anonymous upload is deprecated. Please use authenticated endpoint.", DeprecationWarning)
    
    if not file.filename.lower().endswith('.pdf'):
        return {"error": "Only PDF files are supported"}
    
    # Save uploaded file temporarily
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)  
    file_path = os.path.join(upload_dir, f"{uuid4().hex}_{file.filename}")
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the PDF
        result = process_pdf_internal(file_path)
        
        return result
        
    except Exception as e:
        return {"error": f"Upload failed: {str(e)}"}


def process_pdf_internal(file_path: str):
    """Internal function to process PDF without rate limiting"""
    global VECTOR_STORE, DOCUMENT_REGISTRY, DOCUMENT_EMBEDDINGS

    if not os.path.exists(file_path):
        return {"error": "File not found."}

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)


    doc_id = str(uuid4())
    filename = os.path.basename(file_path)

    for chunk in chunks:
        chunk.metadata = {
            "doc_id": doc_id,
            "filename": filename
        }



class CompareRequest(BaseModel):
    session_id: str

# -------------------------------------------------------------------
# SESSION CLEANUP
# -------------------------------------------------------------------


# ===============================
# PROCESS PDF (MULTI-DOC SUPPORT + RATE LIMITING)
# ===============================
@app.post("/process-pdf")
@limiter.limit("15/15 minutes")
def process_pdf(
    request: Request, 
    data: PDFPath,
    current_user: User = Depends(require_upload_permission)
):
    """Process PDF from file path (Authentication required)"""
    result = process_pdf_internal(data.filePath)
    
    # Add user info to the result for audit trail
    if result and "doc_id" in result:
        result["processed_by"] = current_user.username
        result["user_id"] = current_user.id
    
    return result


# ===============================
# LIST DOCUMENTS
# ===============================
@app.get("/documents")
def list_documents(current_user: User = Depends(require_view_documents_permission)):
    """List all processed documents (Authentication required)"""
    return {
        "documents": DOCUMENT_REGISTRY,
        "requested_by": current_user.username
    }


# ===============================
# PROCESS DOCUMENT
# ===============================
@app.get("/similarity-matrix")
def similarity_matrix(current_user: User = Depends(require_view_documents_permission)):
    """Get similarity matrix between documents (Authentication required)"""
    if len(DOCUMENT_EMBEDDINGS) < 2:
        return {"error": "At least 2 documents required."}

    doc_ids = list(DOCUMENT_EMBEDDINGS.keys())
    vectors = np.array([DOCUMENT_EMBEDDINGS[d] for d in doc_ids])
    sim_matrix = cosine_similarity(vectors)

    result = {}
    for i, doc_id in enumerate(doc_ids):
        result[doc_id] = {}
        for j, other_id in enumerate(doc_ids):
            result[doc_id][other_id] = float(sim_matrix[i][j])

    return {
        "similarity_matrix": result,
        "requested_by": current_user.username
    }


# ===============================
# ASK QUESTION (MULTI-DOC FILTER + RATE LIMITING)
# ===============================
@app.post("/ask")
@limiter.limit("60/15 minutes")
def ask_question(
    request: Request, 
    data: Question,
    current_user: User = Depends(require_ask_permission)
):
    global VECTOR_STORE

    if VECTOR_STORE is None:
        return {"answer": "Please upload at least one PDF first!"}

    docs = VECTOR_STORE.similarity_search(data.question, k=10)

    if data.doc_ids:
        docs = [d for d in docs if d.metadata.get("doc_id") in data.doc_ids]

    if not docs:
        return {"answer": "No relevant context found."}

    context = "\n\n".join([d.page_content for d in docs])

    if data.doc_ids and len(data.doc_ids) > 1:
        prompt = (
            "You are an AI assistant comparing multiple documents.\n"
            "Clearly structure your answer as:\n"
            "- Similarities\n"
            "- Differences\n"
            "- Unique points per document\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {data.question}\n"
            "Answer:"
        )
    else:
        prompt = (
            "You are a helpful assistant answering questions about a PDF.\n"
            "Use ONLY the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {data.question}\n"
            "Answer:"
        )

    answer = generate_response(prompt, max_new_tokens=300)
    return {"answer": answer}


# ===============================
# SUMMARIZE (MULTI-DOC + RATE LIMITING)
# ===============================
@app.post("/summarize")
@limiter.limit("15/15 minutes")
def summarize_pdf(
    request: Request, 
    data: SummarizeRequest,
    current_user: User = Depends(require_summarize_permission)
):
    global VECTOR_STORE

    if VECTOR_STORE is None:
        return {"summary": "Please upload at least one PDF first!"}

    docs = VECTOR_STORE.similarity_search("Summarize the document.", k=12)

    if data.doc_ids:
        docs = [d for d in docs if d.metadata.get("doc_id") in data.doc_ids]

    if not docs:
        return {"summary": "No document context available."}
@app.post("/process")
@limiter.limit("15/15 minutes")
def process_pdf(request: Request, data: PDFPath):
    """
    Process and store PDF with proper cleanup and thread-safe multi-user support.
    """
    try:
        loader = PyPDFLoader(data.filePath)
        raw_docs = loader.load()

        if not raw_docs:
            return {"error": "PDF file is empty or unreadable. Please check your file."}

        # ── Layer 1: normalize at ingestion ──────────────────────────────────────
        cleaned_docs = []
        for doc in raw_docs:
            cleaned_content = normalize_spaced_text(doc.page_content)
            cleaned_docs.append(Document(page_content=cleaned_content, metadata=doc.metadata))

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(cleaned_docs)
        
        if not chunks:
            return {"error": "No text chunks generated from the PDF. Please check your file."}

        # **KEY FIX**: Store per-session with automatic cleanup of old data
        session_id = request.headers.get("X-Session-ID", "default")
        upload_time = datetime.now().isoformat()
        
        # Thread-safe storage (automatically clears old session data)
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        set_session_vectorstore(session_id, vectorstore, upload_time)
        
        return {
            "message": "PDF processed successfully",
            "session_id": session_id,
            "upload_time": upload_time,
            "chunks_created": len(chunks)
        }
            
    except Exception as e:
        return {
            "error": f"PDF processing failed: {str(e)}",
            "details": "Please ensure the file is a valid PDF"
        }


@app.post("/ask")
@limiter.limit("60/15 minutes")
def ask_question(request: Request, data: AskRequest):
    """
    Answer questions using session-specific PDF context with thread-safe access.
    """
    session_id = request.headers.get("X-Session-ID", "default")
    vectorstore, upload_time = get_session_vectorstore(session_id)
    
    if vectorstore is None:
        return {"answer": "Please upload a PDF first!"}
    
    try:
        # Thread-safe vectorstore access
        with sessions_lock:
            question = data.question
            history = data.history
            conversation_context = ""
            
            if history:
                for msg in history[-5:]:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role and content:
                        conversation_context += f"{role}: {content}\n"
            
            # Search only within current session's vectorstore
            docs = vectorstore.similarity_search(question, k=4)
            if not docs:
                return {"answer": "No relevant context found in the current PDF."}

            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""You are a helpful assistant answering questions ONLY from the provided PDF document.

Conversation History (for context only):
{conversation_context}

Document Context (ONLY reference this):
{context}

Current Question:
{question}

Instructions:
- Answer ONLY using the document context provided above.
- Do NOT use any information from previous documents or conversations outside this context.
- If the answer is not in the document, say so briefly.
- Keep the answer concise (2-3 sentences max).

Answer:"""

            raw_answer = generate_response(prompt, max_new_tokens=512)
            answer = normalize_answer(raw_answer)
            return {"answer": answer}
            
    except Exception as e:
        return {"answer": f"Error processing question: {str(e)}"}

    return {"answer": normalize_answer(answer), "confidence_score": 85}

@app.post("/summarize")
@limiter.limit("15/15 minutes")
def summarize_pdf(request: Request, data: SummarizeRequest):
    """
    Summarize PDF using session-specific context with thread-safe access.
    """
    session_id = request.headers.get("X-Session-ID", "default")
    vectorstore, upload_time = get_session_vectorstore(session_id)
    
    if vectorstore is None:
        return {"summary": "Please upload a PDF first!"}

    try:
        # Thread-safe vectorstore access
        with sessions_lock:
            docs = vectorstore.similarity_search("Give a concise summary of the document.", k=6)
            if not docs:
                return {"summary": "No document context available to summarize."}

            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = (
                "You are a document summarization assistant working with a certificate or official document.\n"
                "RULES:\n"
                "1. Summarize in 6-8 concise bullet points.\n"
                "2. Clearly distinguish: who received the certificate, what course, which company issued it,\n"
                "   who signed it, on what platform, and on what date.\n"
                "3. Return clean, properly formatted text — no character spacing, proper Title Case for names.\n"
                "4. Use ONLY the information in the context below.\n"
                "5. DO NOT reference any other documents or previous PDFs.\n\n"
                f"Context:\n{context}\n\n"
                "Summary (bullet points):"
            )

            raw_summary = generate_response(prompt, max_new_tokens=512)
            summary = normalize_answer(raw_summary)
            return {"summary": summary}
            
    except Exception as e:
        return {"summary": f"Error summarizing PDF: {str(e)}"}


@app.post("/compare")
def compare_documents(
    data: CompareRequest,
    current_user: User = Depends(require_compare_permission)
):
    """Compare selected documents (Authentication required)"""
    global VECTOR_STORE, DOCUMENT_REGISTRY

    if VECTOR_STORE is None:
        return {"comparison": "Upload documents first."}

    if len(data.doc_ids) < 2:
        return {"comparison": "Select at least 2 documents."}

    # Pull more candidates
    docs = VECTOR_STORE.similarity_search("Main topics and differences.", k=15)

    # Filter safely
    docs = [d for d in docs if d.metadata.get("doc_id") in data.doc_ids]

    if not docs:
        return {"comparison": "No comparable content found."}

    # Limit per document to avoid overload
    grouped = {}
    for d in docs:
        grouped.setdefault(d.metadata["doc_id"], []).append(d.page_content)

    context = ""
    for doc_id in data.doc_ids:
        filename = DOCUMENT_REGISTRY.get(doc_id, {}).get("filename", doc_id)
        content = "\n\n".join(grouped.get(doc_id, [])[:4])
        context += f"\n\nDocument: {filename}\n{content}\n"

    prompt = (
        "You are an expert AI that compares documents.\n"
        "Provide a detailed comparison with:\n"
        "1. Overall Themes\n"
        "2. Key Similarities\n"
        "3. Key Differences\n"
        "4. Unique Strengths per Document\n\n"
        f"{context}\n\n"
        "Comparison:"
    )


@app.get("/status")
def get_pdf_status(request: Request):
    """
    Returns the current PDF session status.
    Useful for debugging and ensuring proper state management.
    """
    session_id = request.headers.get("X-Session-ID", "default")
    
    with sessions_lock:
        if session_id in sessions:
            return {
                "pdf_loaded": True,
                "session_id": session_id,
                "upload_time": sessions[session_id].get("upload_time")
            }
        return {
            "pdf_loaded": False,
            "session_id": session_id,
            "upload_time": None
        }


# -------------------------------------------------------------------
# START SERVER
# -------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)