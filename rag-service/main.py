from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import uvicorn
import torch
import time
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from slowapi import Limiter
from slowapi.util import get_remote_address

load_dotenv()

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Session Management
sessions = {}  # Format: { "session_id": { "vectorstore": FAISS, "last_accessed": float } }
SESSION_TIMEOUT = 3600  # 1 hour

HF_GENERATION_MODEL = os.getenv("HF_GENERATION_MODEL", "google/flan-t5-base")
generation_tokenizer = None
generation_model = None
generation_is_encoder_decoder = False

# Load local embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_generation_model():
    global generation_tokenizer, generation_model, generation_is_encoder_decoder
    if generation_model is not None and generation_tokenizer is not None:
        return generation_tokenizer, generation_model, generation_is_encoder_decoder

    config = AutoConfig.from_pretrained(HF_GENERATION_MODEL)
    generation_is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
    generation_tokenizer = AutoTokenizer.from_pretrained(HF_GENERATION_MODEL)

    if generation_is_encoder_decoder:
        generation_model = AutoModelForSeq2SeqLM.from_pretrained(HF_GENERATION_MODEL)
    else:
        generation_model = AutoModelForCausalLM.from_pretrained(HF_GENERATION_MODEL)

    if torch.cuda.is_available():
        generation_model = generation_model.to("cuda")

    generation_model.eval()
    return generation_tokenizer, generation_model, generation_is_encoder_decoder


def generate_response(prompt: str, max_new_tokens: int) -> str:
    tokenizer, model, is_encoder_decoder = load_generation_model()
    model_device = next(model.parameters()).device

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    encoded = {key: value.to(model_device) for key, value in encoded.items()}
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )

    if is_encoder_decoder:
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return text.strip()

    input_len = encoded["input_ids"].shape[1]
    new_tokens = generated_ids[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()

class PDFPath(BaseModel):
    filePath: str
    session_id: str

class Question(BaseModel):
    question: str
    session_id: str


class SummarizeRequest(BaseModel):
    pdf: str | None = None
    session_id: str

def cleanup_expired_sessions():
    current_time = time.time()
    expired = [sid for sid, data in sessions.items() if current_time - data["last_accessed"] > SESSION_TIMEOUT]
    for sid in expired:
        del sessions[sid]

@app.post("/process-pdf")
@limiter.limit("15/15 minutes")
def process_pdf(data: PDFPath):
    cleanup_expired_sessions()

    loader = PyPDFLoader(data.filePath)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    if not chunks:
            return {"error": "No text chunks generated from the PDF. Please check your file."}
    sessions[data.session_id] = {
        "vectorstore": FAISS.from_documents(chunks, embedding_model),
        "last_accessed": time.time()
    }

    return {"message": "PDF processed successfully"}


@app.post("/ask")
@limiter.limit("60/15 minutes")
def ask_question(data: Question):
    cleanup_expired_sessions()
    
    session_data = sessions.get(data.session_id)
    if not session_data:
        return {"answer": "Session expired or no PDF uploaded for this session!"}
        
    session_data["last_accessed"] = time.time()
    vectorstore = session_data["vectorstore"]

    docs = vectorstore.similarity_search(data.question, k=4)
    if not docs:
        return {"answer": "No relevant context found."}

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = (
        "You are a helpful assistant for question answering over PDF documents. "
        "Use only the provided context. If the context does not contain the answer, say so briefly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {data.question}\n"
        "Answer:"
    )

    answer = generate_response(prompt, max_new_tokens=256)
    return {"answer": answer}


@app.post("/summarize")
@limiter.limit("15/15 minutes")
def summarize_pdf(data: SummarizeRequest):
    cleanup_expired_sessions()

    session_data = sessions.get(data.session_id)
    if not session_data:
        return {"summary": "Session expired or no PDF uploaded for this session!"}
        
    session_data["last_accessed"] = time.time()
    vectorstore = session_data["vectorstore"]

    docs = vectorstore.similarity_search("Give a concise summary of the document.", k=6)
    if not docs:
        return {"summary": "No document context available to summarize."}

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = (
        "Summarize the following document content in 6-8 concise bullet points.\n\n"
        f"Context:\n{context}\n\n"
        "Summary:"
    )

    summary = generate_response(prompt, max_new_tokens=220)
    return {"summary": summary}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
