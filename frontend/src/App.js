import React, { useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { Container } from "react-bootstrap";
import { pdfjs } from "react-pdf";

// Import custom hooks
import { useTheme } from "./hooks/useTheme";
import { useSession } from "./hooks/useSession";

// Import components
import Header from "./components/Header";
import DocumentUploader from "./components/DocumentUploader";
import DocumentSelector from "./components/DocumentSelector";
import ChatInterface from "./components/ChatInterface";
import ComparisonView from "./components/ComparisonView";

// Initialize PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

function App() {
  // ===============================
  // Custom Hooks
  // ===============================
  const { darkMode, toggleTheme } = useTheme();
  const sessionId = useSession();

  // ===============================
  // State Management
  // ===============================
  const [documents, setDocuments] = useState([]); // { name, doc_id, url, ext }
  const [selectedDocIds, setSelectedDocIds] = useState([]);
  const [chatHistory, setChatHistory] = useState([]);


  // ===============================
  // Event Handlers
  // ===============================

  /**
   * Handle successful document upload
   */
  const handleUploadSuccess = (uploadedDoc) => {
    setDocuments((prev) => [...prev, uploadedDoc]);
  };

  /**
   * Handle document selection toggle
   */
  const handleDocumentToggle = (docId) => {
    setChatHistory([]); // Clear chat when selection changes
    setSelectedDocIds((prev) =>
      prev.includes(docId)
        ? prev.filter((id) => id !== docId)
        : [...prev, docId]
    );
  };

  /**
   * Handle new chat message
   */
  const handleChatUpdate = (message) => {
    setChatHistory((prev) => [...prev, message]);
  };

  /**
   * Get selected document names for display
   */
  const selectedDocNames = documents
    .filter((doc) => selectedDocIds.includes(doc.doc_id))
    .map((doc) => doc.name);

  // ===============================
  // Theme & Styling
  // ===============================
  const pageBg = darkMode ? "bg-dark text-light" : "bg-light text-dark";
  const cardClass = darkMode
    ? "text-white border-secondary shadow"
    : "bg-white text-dark border-0 shadow-sm";
  const inputClass = darkMode ? "text-white border-secondary" : "";

  // ===============================
  // Render
  // ===============================
  return (
    <div className={pageBg} style={{ minHeight: "100vh" }}>
      <Header darkMode={darkMode} onThemeToggle={toggleTheme} />

      <Container className="mt-4">
        {/* Document Upload Section */}
        <DocumentUploader
          onUploadSuccess={handleUploadSuccess}
          sessionId={sessionId}
          darkMode={darkMode}
          cardClass={cardClass}
          inputClass={inputClass}
        />

        {/* Document Selection Section */}
        <DocumentSelector
          documents={documents}
          selectedDocIds={selectedDocIds}
          onSelectionChange={handleDocumentToggle}
          cardClass={cardClass}
        />

        {/* Comparison View (only shown when 2 docs selected) */}
        {selectedDocIds.length === 2 && (
          <ComparisonView
            selectedDocNames={selectedDocNames}
            selectedDocIds={selectedDocIds}
            sessionId={sessionId}
            cardClass={cardClass}
          />
        )}

        {/* Chat Interface (only shown when docs are selected and not comparing) */}
        {selectedDocIds.length !== 2 && selectedDocIds.length > 0 && (
          <ChatInterface
            chatHistory={chatHistory}
            selectedDocIds={selectedDocIds}
            selectedDocCount={selectedDocIds.length}
            sessionId={sessionId}
            cardClass={cardClass}
            inputClass={inputClass}
            onChatUpdate={handleChatUpdate}
          />
        )}

        {/* Empty State Message */}
        {documents.length === 0 && (
          <div className={`text-center mt-5 p-4 ${cardClass}`}>
            <p className="text-muted">
              Start by uploading a PDF or document above
            </p>
          </div>
        )}
      </Container>
    </div>
  );
}

export default App;