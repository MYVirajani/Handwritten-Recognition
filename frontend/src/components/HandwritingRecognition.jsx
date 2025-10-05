// src/components/HandwritingRecognition.jsx
import React, { useState } from 'react';
import { Upload, FileText, Download, Loader2, X, AlertCircle, CheckCircle } from 'lucide-react';
import '../styles/HandwritingRecognition.css';

const HandwritingRecognition = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [recognizedText, setRecognizedText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      const fileType = selectedFile.type;
      const fileSize = selectedFile.size / 1024 / 1024; // in MB

      if (fileSize > 10) {
        setError('File size should not exceed 10MB');
        return;
      }

      if (fileType.startsWith('image/') || fileType === 'application/pdf') {
        setFile(selectedFile);
        setError('');
        setSuccess('');
        setRecognizedText('');
        
        // Preview for images only
        if (fileType.startsWith('image/')) {
          const reader = new FileReader();
          reader.onloadend = () => setPreview(reader.result);
          reader.readAsDataURL(selectedFile);
        } else {
          setPreview(null);
        }
      } else {
        setError('Please upload an image (JPG, PNG) or PDF file');
        setFile(null);
        setPreview(null);
      }
    }
  };

  const handleRemoveFile = () => {
    setFile(null);
    setPreview(null);
    setRecognizedText('');
    setError('');
    setSuccess('');
    document.getElementById('fileInput').value = '';
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      setError('Please select a file to upload');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/api/recognize', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setRecognizedText(data.recognized_text);
        setSuccess('Text recognized successfully!');
      } else {
        setError(data.error || 'Failed to recognize text');
      }
    } catch (err) {
      setError('Failed to connect to server. Please ensure Flask server is running on port 5000.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPDF = async () => {
    if (!recognizedText) return;

    try {
      const response = await fetch('http://localhost:5000/api/download-pdf', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: recognizedText }),
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `recognized_text_${Date.now()}.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        setSuccess('PDF downloaded successfully!');
      } else {
        setError('Failed to download PDF');
      }
    } catch (err) {
      setError('Failed to download PDF. Please try again.');
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      const fakeEvent = { target: { files: [droppedFile] } };
      handleFileChange(fakeEvent);
    }
  };

  return (
    <div className="app-container">
      <div className="header">
        <FileText className="header-icon" size={40} />
        <h1>Handwritten Text Recognition</h1>
        <p>Upload your handwritten images or PDFs and get recognized text instantly</p>
      </div>

      <div className="main-content">
        {/* Upload Section */}
        <div className="upload-section">
          <form onSubmit={handleSubmit}>
            <div 
              className="upload-area"
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              {!file ? (
                <>
                  <Upload className="upload-icon" size={48} />
                  <h3>Drag & Drop your file here</h3>
                  <p>or</p>
                  <label htmlFor="fileInput" className="file-input-label">
                    Browse Files
                  </label>
                  <input
                    id="fileInput"
                    type="file"
                    accept="image/*,.pdf"
                    onChange={handleFileChange}
                    className="file-input"
                  />
                  <p className="file-info">Supported formats: JPG, PNG, PDF (Max 10MB)</p>
                </>
              ) : (
                <div className="file-preview">
                  {preview ? (
                    <img src={preview} alt="Preview" className="preview-image" />
                  ) : (
                    <div className="pdf-preview">
                      <FileText size={64} />
                      <p>PDF File</p>
                    </div>
                  )}
                  <div className="file-details">
                    <p className="file-name">{file.name}</p>
                    <p className="file-size">{(file.size / 1024).toFixed(2)} KB</p>
                  </div>
                  <button
                    type="button"
                    onClick={handleRemoveFile}
                    className="remove-file-btn"
                  >
                    <X size={20} /> Remove
                  </button>
                </div>
              )}
            </div>

            {file && (
              <button
                type="submit"
                disabled={loading}
                className="submit-btn"
              >
                {loading ? (
                  <>
                    <Loader2 className="spinner" size={20} />
                    Processing...
                  </>
                ) : (
                  <>
                    <FileText size={20} />
                    Recognize Text
                  </>
                )}
              </button>
            )}
          </form>

          {/* Error Message */}
          {error && (
            <div className="message error-message">
              <AlertCircle size={20} />
              <span>{error}</span>
            </div>
          )}

          {/* Success Message */}
          {success && (
            <div className="message success-message">
              <CheckCircle size={20} />
              <span>{success}</span>
            </div>
          )}
        </div>

        {/* Results Section */}
        {recognizedText && (
          <div className="results-section">
            <div className="results-header">
              <h2>Recognized Text</h2>
              <button onClick={handleDownloadPDF} className="download-btn">
                <Download size={20} />
                Download as PDF
              </button>
            </div>
            <div className="text-output">
              <pre>{recognizedText}</pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default HandwritingRecognition;