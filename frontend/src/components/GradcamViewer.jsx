import React, { useState } from "react";
import Modal from "./Modal";

export default function GradcamViewer({ originalImageSrc, explanation, isLoading = false }) {
  const [activeViewMode, setActiveViewMode] = useState("columns"); // 'columns' | 'panel'
  const [modalImage, setModalImage] = useState(null);

  if (isLoading) {
    return (
      <div className="gradcam-card" style={{ textAlign: "center", padding: "3rem 1.5rem" }}>
        <div className="spinner-medical" style={{ margin: "0 auto" }} />
        <h4 style={{ fontSize: "1rem", fontWeight: "700", marginTop: "1.25rem", color: "var(--text-primary)" }}>
          Generating Grad-CAM Explainability Heatmaps
        </h4>
        <p style={{ marginTop: "0.35rem", color: "var(--text-muted)", fontSize: "0.85rem" }}>
          Computing gradients of predicted class with respect to 'top_activation' convolutional layer (7&times;7&times;1280)...
        </p>
      </div>
    );
  }

  if (!explanation) return null;

  const { gradcam_overlay_base64, gradcam_heatmap_base64, gradcam_panel_base64 } = explanation;

  const columnsData = [
    {
      id: "original",
      title: "1. Original MRI",
      subtitle: "Input Scan (224×224 RGB)",
      src: originalImageSrc,
      desc: "Unmodified input brain MRI scan after deterministic letterbox preprocessing.",
    },
    {
      id: "heatmap",
      title: "2. Activation Heatmap",
      subtitle: "Convolutional Feature Weights",
      src: gradcam_heatmap_base64,
      desc: "Normalized gradient-weighted feature map from 'top_activation' layer.",
    },
    {
      id: "overlay",
      title: "3. Grad-CAM Overlay",
      subtitle: "Attribution Composite",
      src: gradcam_overlay_base64,
      desc: "60% MRI scan blended with 40% JET colormap activation map.",
    },
  ];

  return (
    <div className="gradcam-card">
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1.25rem", flexWrap: "wrap", gap: "0.75rem" }}>
        <div>
          <h3 className="card-title">Model Explanation</h3>
          <p className="card-subtitle">Regions contributing to the model's prediction</p>
        </div>

        {/* View Toggle & Enlarge Action */}
        <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", flexWrap: "wrap" }}>
          <div style={{ display: "flex", gap: "0.35rem", backgroundColor: "var(--surface-secondary)", padding: "0.25rem", borderRadius: "var(--radius-md)", border: "1px solid var(--border)" }}>
            <button
              className={`btn btn-sm ${activeViewMode === "columns" ? "btn-secondary" : ""}`}
              style={{ padding: "0.35rem 0.65rem", fontSize: "0.78rem" }}
              onClick={() => setActiveViewMode("columns")}
            >
              3-Column View
            </button>
            <button
              className={`btn btn-sm ${activeViewMode === "panel" ? "btn-secondary" : ""}`}
              style={{ padding: "0.35rem 0.65rem", fontSize: "0.78rem" }}
              onClick={() => setActiveViewMode("panel")}
            >
              Full 3-Panel Figure
            </button>
          </div>

          <button
            className="btn btn-secondary btn-sm"
            onClick={() => setModalImage({ title: "Complete 3-Panel Grad-CAM Figure", src: gradcam_panel_base64, desc: "Presentation-ready multi-panel visualization with score card." })}
            title="Open Fullscreen View"
          >
            🔍 Enlarge Full Figure
          </button>
        </div>
      </div>

      {/* 3-Column Desktop View (Strictly Horizontal) */}
      {activeViewMode === "columns" ? (
        <div className="gradcam-columns-grid">
          {columnsData.map((col) => (
            <div
              key={col.id}
              className="gradcam-col-card"
              onClick={() => setModalImage({ title: col.title, src: col.src || originalImageSrc, desc: col.desc })}
              title="Click to enlarge"
            >
              <div className="gradcam-col-header">
                <div className="gradcam-col-title">{col.title}</div>
                <div className="gradcam-col-subtitle">{col.subtitle}</div>
              </div>

              <div className="gradcam-col-image-wrapper">
                <img
                  src={col.src || originalImageSrc}
                  alt={col.title}
                  loading="lazy"
                />
                <div className="gradcam-zoom-hint">🔍 Enlarge</div>
              </div>

              <div className="gradcam-col-caption">
                {col.desc}
              </div>
            </div>
          ))}
        </div>
      ) : (
        /* Full 3-Panel Figure Mode */
        <div
          className="gradcam-view-panel"
          onClick={() => setModalImage({ title: "Complete 3-Panel Figure", src: gradcam_panel_base64, desc: "Presentation-ready multi-panel visualization with score card." })}
        >
          <img
            src={gradcam_panel_base64}
            alt="Complete 3-Panel Grad-CAM Figure"
            style={{ width: "100%", maxHeight: "420px", objectFit: "contain" }}
          />
          <div className="gradcam-zoom-hint">Click to enlarge 🔍</div>
        </div>
      )}

      {/* Mandatory Interpretability Caveat (Section 14) */}
      <div className="gradcam-explanation-text">
        <strong>Interpretability Notice:</strong> Grad-CAM is an interpretability visualization showing image regions that contributed to the model's prediction. It should not be interpreted as a clinically confirmed tumor location.
      </div>

      {/* Modal Zoom */}
      {modalImage && (
        <Modal
          isOpen={Boolean(modalImage)}
          onClose={() => setModalImage(null)}
          title={modalImage.title}
        >
          <img
            src={modalImage.src}
            alt={modalImage.title}
            style={{ maxWidth: "100%", maxHeight: "56vh", objectFit: "contain" }}
          />
          <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginTop: "1rem", textAlign: "center" }}>
            {modalImage.desc}
          </p>
          <div style={{ marginTop: "1.25rem", textAlign: "center" }}>
            <button className="btn btn-secondary btn-sm" onClick={() => setModalImage(null)}>
              Close Fullscreen View
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}
