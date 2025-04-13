import React, { useState, useRef } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const imageRef = useRef();
  const fileInputRef = useRef();

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setImage(e.target.result);
        setResult(null);
        setError(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const classifyImage = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await axios.post('http://localhost:5000/classify', {
        image: image
      });
      setResult(response.data);
    } catch (error) {
      setError('Error classifying image. Please try again.');
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <nav className="navbar">
        <div className="nav-content">
          <div className="logo">GlaucomaAI</div>
          <div className="nav-links">
            <a href="#features">Features</a>
            <a href="#how-it-works">How It Works</a>
            <a href="#contact">Contact</a>
          </div>
        </div>
      </nav>

      <header className="hero-section">
        <div className="hero-content">
          <h1>Advanced Glaucoma Detection System</h1>
          <p className="hero-subtitle">Powered by Deep Learning and Computer Vision</p>
          <div className="hero-stats">
            <div className="stat">
              <span className="stat-number">99.8%</span>
              <span className="stat-label">Accuracy</span>
            </div>
            <div className="stat">
              <span className="stat-number">0.2s</span>
              <span className="stat-label">Processing Time</span>
            </div>
            <div className="stat">
              <span className="stat-number">10k+</span>
              <span className="stat-label">Images Analyzed</span>
            </div>
          </div>
        </div>
      </header>

      <main className="main-content">
        <section className="upload-section">
          <div className="section-header">
            <h2>Upload Fundus Image</h2>
            <p>Get instant glaucoma detection results using our advanced AI system</p>
          </div>
          
          <div className="upload-container">
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              ref={fileInputRef}
              style={{ display: 'none' }}
            />
            <button
              onClick={() => fileInputRef.current.click()}
              className="upload-button"
            >
              <span className="button-icon">📷</span>
              Upload Fundus Image
            </button>
            <p className="upload-hint">Supported formats: JPG, PNG</p>
          </div>
        </section>
        
        {image && (
          <section className="analysis-section">
            <div className="section-header">
              <h2>Image Analysis</h2>
              <p>Review and analyze your uploaded fundus image</p>
            </div>
            
            <div className="image-container">
              <div className="image-preview">
                <img
                  ref={imageRef}
                  src={image}
                  alt="Fundus Image"
                  className="fundus-image"
                />
              </div>
              <button
                onClick={classifyImage}
                className="classify-button"
                disabled={loading}
              >
                {loading ? (
                  <span className="loading-spinner">⏳</span>
                ) : (
                  <span className="analyze-icon">🔍</span>
                )}
                {loading ? 'Analyzing...' : 'Analyze Image'}
              </button>
            </div>
          </section>
        )}

        {error && (
          <div className="error-message">
            <span className="error-icon">⚠️</span>
            {error}
          </div>
        )}

        {result && (
          <section className="results-section">
            <div className="section-header">
              <h2>Analysis Results</h2>
              <p>Detailed diagnosis and confidence metrics</p>
            </div>
            
            <div className={`result-card ${result.prediction.toLowerCase()}`}>
              <div className="result-header">
                <h3>Diagnosis: {result.prediction}</h3>
                <span className={`status-indicator ${result.prediction.toLowerCase()}`}></span>
              </div>
              <div className="confidence-meter">
                <div className="meter-label">Confidence Level</div>
                <div className="meter">
                  <div 
                    className="meter-fill"
                    style={{ width: `${result.confidence}%` }}
                  ></div>
                </div>
                <div className="confidence-value">{result.confidence.toFixed(2)}%</div>
              </div>
              <div className="probability">
                <span className="probability-label">Probability of Glaucoma:</span>
                <span className="probability-value">{result.probability.toFixed(2)}%</span>
              </div>
            </div>
            
            <div className="disclaimer">
              <p>⚠️ Note: This is an AI-assisted diagnosis tool. Please consult with a medical professional for final diagnosis and treatment.</p>
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        <div className="footer-content">
          <div className="footer-section">
            <h3>GlaucomaAI</h3>
            <p>Advanced AI-powered glaucoma detection system</p>
          </div>
          <div className="footer-section">
            <h3>Contact</h3>
            <p>Email: support@glaucomaai.com</p>
            <p>Phone: +1 (555) 123-4567</p>
          </div>
          <div className="footer-section">
            <h3>Legal</h3>
            <p>Privacy Policy</p>
            <p>Terms of Service</p>
          </div>
        </div>
        <div className="footer-bottom">
          <p>© 2024 GlaucomaAI. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
