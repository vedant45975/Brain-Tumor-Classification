import React from "react";
import { useNavigate } from "react-router-dom";
import "./Dashboard.css";

function Dashboard() {
  const navigate = useNavigate();
  const username = localStorage.getItem("username");
  const hospital = localStorage.getItem("hospital");

  const handleLogout = () => {
    localStorage.removeItem("username");
    localStorage.removeItem("hospital");
    navigate("/", { replace: true });
  };

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <div className="header-left">
          <h1>🧠 Brain Tumor Classification System</h1>
          <p className="subtitle">Advanced AI-Powered Medical Imaging Analysis</p>
        </div>
        <div className="header-right">
          <div className="user-info">
            <p><strong>User:</strong> {username}</p>
            {hospital && <p><strong>Hospital:</strong> {hospital}</p>}
          </div>
          <button className="logout-btn" onClick={handleLogout}>Logout</button>
        </div>
      </div>

      <div className="dashboard-content">
        <div className="models-grid">
          <div className="model-card" onClick={() => navigate("/qml")}>
            <div className="card-icon">⚛️</div>
            <h2>Quantum ML Model</h2>
            <p className="model-description">
              Advanced Quantum Machine Learning for brain tumor classification
            </p>
            <ul className="model-features">
              <li>✓ Quantum-enhanced predictions</li>
              <li>✓ Higher accuracy</li>
              <li>✓ Advanced pattern recognition</li>
            </ul>
            <button className="card-btn">Use QML Analysis →</button>
          </div>

          <div className="model-card" onClick={() => navigate("/cml")}>
            <div className="card-icon">🧮</div>
            <h2>Classical ML Model</h2>
            <p className="model-description">
              Traditional CNN-based neural network for tumor classification
            </p>
            <ul className="model-features">
              <li>✓ Fast processing</li>
              <li>✓ Reliable results</li>
              <li>✓ Real-time analysis</li>
            </ul>
            <button className="card-btn">Use CNN Analysis →</button>
          </div>
        </div>

        <div className="info-section">
          <h3>How to Use:</h3>
          <ol>
            <li>Select either Quantum ML or Classical ML model</li>
            <li>Upload a brain MRI image (JPEG, PNG, etc.)</li>
            <li>Click "Predict" to analyze the image</li>
            <li>View the tumor type and confidence score</li>
            <li>Download or save results if needed</li>
          </ol>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
