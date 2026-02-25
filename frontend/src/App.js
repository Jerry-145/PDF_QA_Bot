import React, { useState, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import "bootstrap/dist/css/bootstrap.min.css";
import {
  Container,
  Button,
  Form,
  Card,
  Spinner,
  Navbar,
  Dropdown

} from "react-bootstrap";

const API_BASE = process.env.REACT_APP_API_URL || "";
const THEME_STORAGE_KEY = "pdf-qa-bot-theme";

function App() {
  // Core state
  const [file, setFile] = useState(null);
  // Multi-document support (from remote)
  const [pdfs, setPdfs] = useState([]); // {name, doc_id, url, chat: [], processed: false}
  const [selectedDocs, setSelectedDocs] = useState([]);
  const [question, setQuestion] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [comparisonResult, setComparisonResult] = useState(null);
  
  // Single document support (from local)
  const [selectedPdf, setSelectedPdf] = useState(null);
  
  // Common state
  const [question, setQuestion] = useState("");
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [processingPdf, setProcessingPdf] = useState(false); // Track PDF processing status
  const [summarizing, setSummarizing] = useState(false);
  const [comparing, setComparing] = useState(false);
  
  // Theme management (from local) 
  const [darkMode, setDarkMode] = useState(() => {
    // Load theme preference from localStorage
    const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
    return savedTheme ? JSON.parse(savedTheme) : false;
  });
  
  // PDF viewer state (from local)
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);

  // Save theme preference to localStorage
  useEffect(() => {
    localStorage.setItem(THEME_STORAGE_KEY, JSON.stringify(darkMode));
  }, [darkMode]);

  // ===============================
  // Upload - Combined approach
  // ===============================
  const uploadPDF = async () => {
    if (!file) return;

    setUploading(true);
    setProcessingPdf(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      // Upload and process PDF

  // Theme
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    return saved ? JSON.parse(saved) : false;
  });

  useEffect(() => {
    localStorage.setItem(THEME_STORAGE_KEY, JSON.stringify(darkMode));
  }, [darkMode]);

  // Upload
  const uploadDocument = async () => {
    if (!file) return;

    setUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await axios.post(`${API_BASE}/upload`, formData);

      const url = URL.createObjectURL(file);
      
      // Add PDF with both doc_id (remote) and chat history (local)
      const newPdf = { 
        name: file.name, 
        doc_id: res.data.doc_id, 
        url, 
        chat: [], 
        processed: true 
      };
      
      setPdfs(prev => [...prev, newPdf]);
      setSelectedPdf(file.name); // Set as selected for single-doc mode
      setFile(null);
      setProcessingPdf(false);
      alert("PDF uploaded and processed successfully!");
    } catch (e) {
      const message = e.response?.data?.error || "Upload failed.";
      alert(message);
      setProcessingPdf(false);
    }
  };

  const toggleDocSelection = (doc_id) => {
    setComparisonResult(null);
    setSelectedDocs((prev) =>
      prev.includes(sessionId)
        ? prev.filter((id) => id !== sessionId)
        : [...prev, sessionId]
    );
  };

  // ===============================
  // Ask Question - Multi-doc mode
  // ===============================
  const askQuestion = async () => {
    if (!question.trim() || selectedDocs.length === 0) return;

    setAsking(true);
    setChatHistory((prev) => [...prev, { role: "user", text: question }]);

    try {
      // Use the first selected document's sessionId
      const res = await axios.post(`${API_BASE}/ask`, {
        question,
        sessionId: selectedDocs[0], // Send the stored sessionId
      });

      setChatHistory((prev) => [
        ...prev,
        {
          role: "bot",
          text: res.data.answer,
        },
      ]);
    } catch {
      setChatHistory((prev) => [
        ...prev,
        { role: "bot", text: "Error getting answer." },
      ]);
    } finally {
      setQuestion("");
      setAsking(false);
    }
  };

  // ===============================
  // Ask Question - Single-doc mode  
  // ===============================
  const askQuestionSingle = async () => {
    if (!question.trim() || !selectedPdf) return;

    const pdfData = pdfs.find(pdf => pdf.name === selectedPdf);
    if (!pdfData || !pdfData.processed) return;

    // Add question to single PDF's chat history
    const updatedChat = [...pdfData.chat, { role: "user", text: question }];
    setPdfs(prev => prev.map(pdf => 
      pdf.name === selectedPdf 
        ? { ...pdf, chat: updatedChat }
        : pdf
    ));

    setAsking(true);

    try {
      const res = await axios.post(`${API_BASE}/ask`, {
        question,
        doc_ids: [pdfData.doc_id]
      });

      // Add answer to chat
      setPdfs(prev => prev.map(pdf => 
        pdf.name === selectedPdf 
          ? { ...pdf, chat: [...updatedChat, { role: "bot", text: res.data.answer }] }
          : pdf
      ));
    } catch {
      setPdfs(prev => prev.map(pdf => 
        pdf.name === selectedPdf 
          ? { ...pdf, chat: [...updatedChat, { role: "bot", text: "Error getting answer." }] }
          : pdf
      ));
    }

    setQuestion("");
    setAsking(false);
  };

  // ===============================
  // Summarize
  const summarizePDF = async () => {
    if (selectedDocs.length === 0 && !selectedPdf) return;

    setSummarizing(true);
    try {
      let doc_ids;
      if (selectedDocs.length > 0) {
        // Multi-doc mode
        doc_ids = selectedDocs;
      } else {
        // Single-doc mode
        const pdfData = pdfs.find(pdf => pdf.name === selectedPdf);
        doc_ids = [pdfData.doc_id];
      }

      const res = await axios.post(`${API_BASE}/summarize`, { doc_ids });

      if (selectedDocs.length > 0) {
        // Add to multi-doc chat
        setChatHistory(prev => [
          ...prev,
          { role: "bot", text: res.data.summary }
        ]);
      } else {
        // Add to single-doc chat
        setPdfs(prev => prev.map(pdf => 
          pdf.name === selectedPdf 
            ? { ...pdf, chat: [...pdf.chat, { role: "bot", text: res.data.summary }] }
            : pdf
        ));
      }
    } catch {
      alert("Error summarizing.");
    } finally {
      setSummarizing(false);
    }
  };

  // Compare
  const compareDocuments = async () => {
    if (selectedDocs.length < 2) return;

    setComparing(true);
    try {
      const res = await axios.post(`${API_BASE}/compare`, {
        sessionId,
        doc_ids: selectedDocs,
      });

      setComparisonResult(res.data.comparison);

      setChatHistory((prev) => [
        ...prev,
        { role: "bot", text: res.data.comparison },
      ]);
    } catch {
      alert("Error comparing.");
    } finally {
      setComparing(false);
    }
  };

  // ===============================
  // Export Functions (from local)
  // ===============================
  const exportChat = (format) => {
    const currentChat = pdfs.find(pdf => pdf.name === selectedPdf)?.chat || [];
    if (currentChat.length === 0) return;

    if (format === "csv") {
      const csvContent = [
        ["Role", "Message"],
        ...currentChat.map(msg => [msg.role, msg.text.replace(/"/g, '""')])
      ].map(row => `"${row.join('","')}"`).join('\n');

      const blob = new Blob([csvContent], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${selectedPdf}_chat.csv`;
      a.click();
    }
    // Add PDF export functionality here if needed
  };

  // Helper variables
  const selectedPdfs = pdfs.filter(p =>
    selectedDocs.includes(p.doc_id)
  );

  const themeClass = darkMode ? "bg-dark text-light" : "bg-light text-dark";
  const currentChat = pdfs.find(pdf => pdf.name === selectedPdf)?.chat || [];
  const currentPdfUrl = pdfs.find(pdf => pdf.name === selectedPdf)?.url || null;
  const isPdfProcessed = pdfs.find(pdf => pdf.name === selectedPdf)?.processed || false;
  const canAskQuestions = selectedPdf && isPdfProcessed && !processingPdf;

  const currentChat = pdfs.find(pdf => pdf.name === selectedPdf)?.chat || [];
  const currentPdfUrl = pdfs.find(pdf => pdf.name === selectedPdf)?.url || null;
  const isPdfProcessed = pdfs.find(pdf => pdf.name === selectedPdf)?.processed || false;
  const canAskQuestions = selectedPdf && isPdfProcessed && !processingPdf;
  const selectedPdfs = pdfs.filter((p) =>
    selectedDocs.includes(p.sessionId)
  );

  return (
    <div className={darkMode ? "bg-dark text-light" : "bg-light"}>
      <Navbar bg="primary" variant="dark">
        <Container className="d-flex justify-content-between">
          <Navbar.Brand>PDF Q&A Bot</Navbar.Brand>
          <Button variant="outline-light" onClick={() => setDarkMode(!darkMode)}>
            {darkMode ? "☀️ Light" : "🌙 Dark"} Mode
          </Button>
        </Container>
      </Navbar>

      <Container className="mt-4">
        <Card className="mb-4">
          <Card.Body>
            <h5>📤 Upload PDF</h5>
            <Form>
              <Form.Control 
                type="file" 
                onChange={e => setFile(e.target.files[0])}
                className={darkMode ? "bg-dark text-light border-secondary" : ""}
              />
              <Button
                className="mt-2"
                onClick={uploadPDF}
                disabled={!file || uploading || processingPdf}
              >
                {uploading || processingPdf ? <Spinner size="sm" animation="border" /> : "Upload"}
              </Button>
              {processingPdf && <div className="mt-2 text-info small"><Spinner animation="border" size="sm" className="me-2" />Processing PDF...</div>}
              {file && <span className="ms-3 text-muted">{file.name}</span>}
            </Form>

            {/* PDF Selection for Single-Doc Mode */}
            {pdfs.length > 0 && (
              <Dropdown className="mt-3">
                <Dropdown.Toggle variant={darkMode ? "outline-light" : "info"} id="dropdown-pdf">
                  📚 {selectedPdf || "Select PDF (Single Mode)"}
                </Dropdown.Toggle>
                <Dropdown.Menu className={darkMode ? "bg-dark" : ""}>
                  {pdfs.map(pdf => (
                    <Dropdown.Item
                      key={pdf.name}
                      onClick={() => setSelectedPdf(pdf.name)}
                      className={darkMode ? "text-light" : ""}
                      style={darkMode ? { backgroundColor: 'transparent' } : {}}
                    >
                      {pdf.name} {pdf.processed ? "✅" : "⏳"}
                    </Dropdown.Item>
                  ))}
                </Dropdown.Menu>
              </Dropdown>
            )}
          </Card.Body>
        </Card>

        {/* Multi-Document Selection */}
        {pdfs.length > 0 && (
          <Card className="mb-4">
            <Card.Body>
              <h5>📋 Multi-Document Mode</h5>
              <p className="text-muted small">Select multiple documents for comparison and cross-document analysis</p>
              {pdfs.map(pdf => (
                <Form.Check
                  key={pdf.doc_id}
                  type="checkbox"
                  label={`${pdf.name} ${pdf.processed ? "✅" : "⏳"}`}
                  checked={selectedDocs.includes(pdf.doc_id)}
                  onChange={() => toggleDocSelection(pdf.doc_id)}
                  className={darkMode ? "text-light" : ""}
                />
              ))}
            </Card.Body>
          </Card>
        )}

        {/* Single Document Chat */}
        {selectedPdf && selectedDocs.length === 0 && (
          <Row>
            <Col md={currentPdfUrl ? 6 : 12}>
              <Card className="mb-4">
                <Card.Body>
                  <h5>💬 Chat with {selectedPdf}</h5>
                  
                  <div style={{ maxHeight: 300, overflowY: "auto", marginBottom: 16 }} className={darkMode ? "border border-secondary p-2 rounded" : "border p-2 rounded"}>
                    {currentChat.length === 0 ? (
                      <div className="text-center text-muted py-4">
                        <p>{canAskQuestions ? "No messages yet. Ask a question about your PDF!" : "Upload and process a PDF to start chatting."}</p>
                      </div>
                    ) : (
                      currentChat.map((msg, i) => (
                        <div key={i} className={`d-flex ${msg.role === "user" ? "justify-content-end" : "justify-content-start"} mb-2`}>
                          <div
                            className={`p-2 rounded ${msg.role === "user"
                                ? "bg-primary text-light"
                                : darkMode
                                  ? "bg-dark text-light border border-secondary"
                                  : "bg-white text-dark border"
                              }`}
                            style={{ maxWidth: "80%" }}
                          >
                            {msg.role === "bot" ? (
                              <div className="markdown-content">
                                <ReactMarkdown>{msg.text}</ReactMarkdown>
                              </div>
                            ) : (
                              <span><strong>You:</strong> {msg.text}</span>
                            )}
                          </div>
                        </div>
                      ))
                    )}
                  </div>

                  {!canAskQuestions && (
                    <div className="alert alert-info mb-3 d-flex align-items-center">
                      <span className="me-2">ℹ️</span>
                      <span>
                        {!selectedPdf ? "Please upload a PDF first to start asking questions." :
                         processingPdf ? "Please wait while your PDF is being processed..." :
                         !isPdfProcessed ? "Your PDF is still being processed. Please wait..." : ""}
                      </span>
                    </div>
                  )}

                  <Form className="d-flex gap-2 mb-3">
                    <Form.Control
                      type="text"
                      placeholder={canAskQuestions ? "Ask a question..." : "Upload a PDF first..."}
                      value={question}
                      onChange={e => setQuestion(e.target.value)}
                      disabled={asking || !canAskQuestions}
                      onKeyDown={e => { if (e.key === "Enter" && canAskQuestions) { e.preventDefault(); askQuestionSingle(); } }}
                      className={darkMode ? "bg-dark text-light border-secondary" : ""}
                    />
                    <Button variant="success" onClick={askQuestionSingle} disabled={asking || !question.trim() || !canAskQuestions}>
                      {asking ? <Spinner animation="border" size="sm" /> : "💭 Ask"}
                    </Button>
                  </Form>

                  <div className="d-flex gap-2 flex-wrap">
                    <Button
                      variant={darkMode ? "outline-warning" : "warning"}
                      onClick={summarizePDF}
                      disabled={summarizing || !canAskQuestions}
                    >
                      {summarizing ? <Spinner animation="border" size="sm" /> : "📝 Summarize PDF"}
                    </Button>
                    <Button
                      variant={darkMode ? "outline-light" : "outline-secondary"}
                      onClick={() => exportChat("csv")}
                      disabled={!selectedPdf || currentChat.length === 0}
                    >
                      📊 Export CSV
                    </Button>
                  </div>
                </Card.Body>
              </Card>
            </Col>

            {/* PDF Viewer */}
            {currentPdfUrl && (
              <Col md={6}>
                <Card>
                  <Card.Body>
                    <h6>📄 PDF Viewer</h6>
                    <Document 
                      file={currentPdfUrl}
                      onLoadSuccess={({ numPages }) => setNumPages(numPages)}
                    >
                      <Page pageNumber={pageNumber} width={400} />
                    </Document>
                    {numPages && (
                      <div className="mt-2 d-flex justify-content-between align-items-center">
                        <Button 
                          size="sm" 
                          onClick={() => setPageNumber(Math.max(1, pageNumber - 1))}
                          disabled={pageNumber <= 1}
                        >
                          Previous
                        </Button>
                        <span>Page {pageNumber} of {numPages}</span>
                        <Button 
                          size="sm" 
                          onClick={() => setPageNumber(Math.min(numPages, pageNumber + 1))}
                          disabled={pageNumber >= numPages}
                        >
                          Next
                        </Button>
                      </div>
                    )}
                  </Card.Body>
                </Card>
              </Col>
            )}
          </Row>
        )}

        {/* Side-by-side View (ONLY when exactly 2 selected in multi-doc mode) */}
        {selectedPdfs.length === 2 && (
          <>
            <Row className="mb-4">
              {selectedPdfs.map(pdf => (
                <Col key={pdf.doc_id} md={6}>
                  <Card>
                    <Card.Body>
                      <h6>📄 {pdf.name}</h6>
                      <Document file={pdf.url}>
                        <Page pageNumber={1} width={300} />
                      </Document>
                    </Card.Body>
                  </Card>
                </Col>
              ))}
            </Row>

            <Card className="mb-4">
              <Card.Body>
                <Button
                  variant="info"
                  onClick={compareDocuments}
                  disabled={comparing}
                >
                  {comparing ? <Spinner size="sm" animation="border" /> : "🔍 Generate Comparison"}
                </Button>

                {comparisonResult && (
                  <div className="mt-4">
                    <h5>🔍 AI Comparison</h5>
                    <ReactMarkdown>{comparisonResult}</ReactMarkdown>
                  </div>
                )}
              </Card.Body>
            </Card>
          </>
        )}

        {/* Multi-Document Chat Mode */}
        {selectedPdfs.length !== 2 && selectedDocs.length > 0 && (
          <Card>
            <Card.Body>
              <h5>💬 Ask Across Selected Documents ({selectedPdfs.length} docs)</h5>

              <div style={{ maxHeight: 300, overflowY: "auto", marginBottom: 16 }} className={darkMode ? "border border-secondary p-2 rounded" : "border p-2 rounded"}>
                {chatHistory.map((msg, i) => (
                  <div key={i} className="mb-2">
                    <strong>{msg.role === "user" ? "You" : "Bot"}:</strong>
                    <ReactMarkdown>{msg.text}</ReactMarkdown>
                  </div>
                ))}
              </div>

              <Form className="d-flex gap-2 mb-3">
                <Form.Control
                  type="text"
                  placeholder="Ask a question across selected documents..."
                  value={question}
                  onChange={e => setQuestion(e.target.value)}
                  className={darkMode ? "bg-dark text-light border-secondary" : ""}
                />
                <Button
                  variant="success"
                  onClick={askQuestion}
                  disabled={asking}
                >
                  {asking ? <Spinner size="sm" animation="border" /> : "Ask"}
                </Button>
              </Form>

              <div className="d-flex gap-2">
                <Button
                  variant="warning"
                  onClick={summarizePDF}
                  disabled={summarizing}
                >
                  {summarizing ? <Spinner size="sm" animation="border" /> : "📝 Summarize"}
                </Button>

                <Button
                  variant="info"
                  onClick={compareDocuments}
                  disabled={selectedDocs.length < 2 || comparing}
                >
                  {comparing ? <Spinner size="sm" animation="border" /> : "🔍 Compare Selected"}
                </Button>
              </div>

            </Card.Body>
          </Card>
        )}

        <div style={{ maxHeight: 300, overflowY: "auto" }}>
          {chatHistory.map((msg, i) => (
            <div key={i}>
              <b>{msg.role}:</b>
              <ReactMarkdown>{msg.text}</ReactMarkdown>
            </div>
          ))}
        </div>

        <Form
          onSubmit={(e) => {
            e.preventDefault();
            askQuestion();
          }}
        >
          <Form.Control
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask..."
          />
        </Form>

        <Button onClick={summarizePDF} disabled={summarizing}>
          {summarizing ? "Summarizing..." : "Summarize"}
        </Button>
      </Container>
    </div>
  );
}

export default App;