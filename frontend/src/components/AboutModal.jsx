import React from "react";
import Modal from "./Modal";
import { DISCLAIMER_TEXT } from "../utils/constants";

export default function AboutModal({ isOpen, onClose }) {
  return (
    <Modal isOpen={isOpen} onClose={onClose} title="About NeuroVision AI Research Platform">
      <div className="about-modal-content">
        {/* Header Summary */}
        <div className="about-modal-header">
          <div className="about-logo-badge">
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
              <rect width="32" height="32" rx="8" fill="#0f172a" />
              <circle cx="16" cy="16" r="10" stroke="#0284c7" strokeWidth="1.75" strokeDasharray="3 2" />
              <path d="M10 16c0-3.3 2.7-6 6-6s6 2.7 6 6-2.7 6-6 6" stroke="#38bdf8" strokeWidth="2" strokeLinecap="round" />
              <circle cx="16" cy="16" r="2.25" fill="#38bdf8" />
            </svg>
          </div>
          <div>
            <h3 style={{ fontSize: "1.25rem", fontWeight: "800", color: "var(--text-primary)" }}>
              NeuroVision AI
            </h3>
            <p style={{ fontSize: "0.875rem", color: "var(--text-muted)" }}>
              Deep Learning Brain MRI Classification & Interpretability Framework
            </p>
          </div>
        </div>

        {/* Specifications Grid */}
        <div className="about-specs-grid">
          <div className="about-spec-item">
            <span className="spec-label">Architecture</span>
            <span className="spec-val font-mono">EfficientNetB0 (4.05M params)</span>
          </div>
          <div className="about-spec-item">
            <span className="spec-label">Input Shape</span>
            <span className="spec-val font-mono">224 × 224 × 3 (Letterboxed)</span>
          </div>
          <div className="about-spec-item">
            <span className="spec-label">Test-Set Accuracy</span>
            <span className="spec-val font-mono text-success">92.81% (1,485 / 1,600)</span>
          </div>
          <div className="about-spec-item">
            <span className="spec-label">Macro F1-Score</span>
            <span className="spec-val font-mono">92.63% (Balanced)</span>
          </div>
          <div className="about-spec-item">
            <span className="spec-label">Dataset Scope</span>
            <span className="spec-val">7,200 Scans (5,600 train / 1,600 test)</span>
          </div>
          <div className="about-spec-item">
            <span className="spec-label">Explainability</span>
            <span className="spec-val">Grad-CAM (top_activation layer)</span>
          </div>
        </div>

        {/* System Architecture Section */}
        <div style={{ marginTop: "1.5rem" }}>
          <h4 style={{ fontSize: "1rem", fontWeight: "700", marginBottom: "0.5rem" }}>
            Classification Target Classes
          </h4>
          <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", lineHeight: "1.6" }}>
            The neural network was trained on four distinct categories:
          </p>
          <ul style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginLeft: "1.25rem", marginTop: "0.5rem", lineHeight: "1.7" }}>
            <li><strong>Glioma:</strong> Intra-axial tumors arising from glial cells within brain parenchyma.</li>
            <li><strong>Meningioma:</strong> Extra-axial, slow-growing tumors arising from the meningeal envelope.</li>
            <li><strong>Pituitary:</strong> Benign adenomas located within the sella turcica / pituitary fossa.</li>
            <li><strong>No Tumor:</strong> Normal brain MRI controls displaying intact neuroanatomy.</li>
          </ul>
        </div>

        {/* Security & Modality Guardrail */}
        <div style={{ marginTop: "1.25rem", padding: "1rem", backgroundColor: "var(--surface-secondary)", borderRadius: "var(--radius-md)", border: "1px solid var(--border)" }}>
          <h4 style={{ fontSize: "0.95rem", fontWeight: "700", color: "var(--text-primary)", display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span>🛡️</span> Mandatory Brain MRI Input Guardrail
          </h4>
          <p style={{ fontSize: "0.825rem", color: "var(--text-secondary)", marginTop: "0.35rem", lineHeight: "1.55" }}>
            Uploaded images are evaluated by a multi-stage modality verification layer (chromatic saturation, perimeter border brightness, cranial morphology) before convolutional inference. Non-MRI images and degraded scans are rejected server-side to prevent erroneous classifications.
          </p>
        </div>

        {/* Mandatory Research Disclaimer */}
        <div style={{ marginTop: "1.25rem", padding: "1rem", backgroundColor: "#fffbeb", borderRadius: "var(--radius-md)", border: "1px solid #fde68a" }}>
          <div style={{ display: "flex", gap: "0.6rem", alignItems: "flex-start" }}>
            <span style={{ fontSize: "1.25rem", lineHeight: 1 }}>⚖️</span>
            <div>
              <strong style={{ fontSize: "0.85rem", color: "#92400e" }}>Academic & Research Notice:</strong>
              <p style={{ fontSize: "0.825rem", color: "#92400e", marginTop: "0.25rem", lineHeight: "1.5" }}>
                {DISCLAIMER_TEXT}
              </p>
            </div>
          </div>
        </div>

        {/* Close Button */}
        <div style={{ marginTop: "1.5rem", textAlign: "right" }}>
          <button className="btn btn-secondary btn-sm" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </Modal>
  );
}
