import React, { useState } from "react";
import StatCard from "../components/StatCard";
import Modal from "../components/Modal";
import { PER_CLASS_PERFORMANCE } from "../utils/constants";

export default function ModelPerformance() {
  const [activeMatrixTab, setActiveMatrixTab] = useState("raw");
  const [modalImage, setModalImage] = useState(null);

  const testSummaryStats = [
    { value: "1,600", label: "Test Set Scans", detail: "400 images per class (Strictly untouched)" },
    { value: "1,485", label: "Correct Predictions", detail: "92.81% test accuracy" },
    { value: "115", label: "Incorrect Predictions", detail: "7.19% test error rate" },
    { value: "92.63%", label: "Macro F1-Score", detail: "Balanced harmonic mean across all 4 classes" },
  ];

  return (
    <div className="container">
      {/* Title */}
      <div style={{ marginBottom: "1.75rem" }}>
        <h1 style={{ fontSize: "2rem", fontWeight: "800" }}>Model Performance & Benchmark Evaluation</h1>
        <p style={{ color: "var(--text-muted)", fontSize: "0.95rem", marginTop: "0.25rem" }}>
          Empirical evaluation results of the EfficientNetB0 classifier measured on the held-out 1,600-image test set.
        </p>
      </div>

      {/* Test Set High-Level Metrics */}
      <section className="metrics-summary-grid">
        {testSummaryStats.map((item, idx) => (
          <StatCard
            key={idx}
            value={item.value}
            label={item.label}
            detail={item.detail}
            highlight={idx === 1}
          />
        ))}
      </section>

      {/* Macro & Weighted Aggregate Metrics Card */}
      <div className="card" style={{ marginBottom: "2rem" }}>
        <h3 className="card-title">Macro & Weighted Aggregate Metrics</h3>
        <p className="card-subtitle">Calculated without sample re-weighting across balanced test classes (N=1,600)</p>

        <div className="macro-metrics-row">
          <div className="macro-metric-cell">
            <div className="macro-metric-title">Test-set classification accuracy</div>
            <div className="macro-metric-val" style={{ color: "var(--primary-accent)" }}>92.81%</div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.2rem" }}>
              1,485 / 1,600 correct
            </div>
          </div>

          <div className="macro-metric-cell">
            <div className="macro-metric-title">Macro Precision</div>
            <div className="macro-metric-val" style={{ color: "#0d9488" }}>93.13%</div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.2rem" }}>
              Unweighted average precision
            </div>
          </div>

          <div className="macro-metric-cell">
            <div className="macro-metric-title">Macro Recall</div>
            <div className="macro-metric-val" style={{ color: "#2563eb" }}>92.81%</div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.2rem" }}>
              Unweighted average recall
            </div>
          </div>

          <div className="macro-metric-cell">
            <div className="macro-metric-title">Macro F1-Score</div>
            <div className="macro-metric-val" style={{ color: "#7c3aed" }}>92.63%</div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.2rem" }}>
              Harmonic mean of precision & recall
            </div>
          </div>
        </div>
      </div>

      {/* Per-Class Performance Table */}
      <section className="card" style={{ marginBottom: "2rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "0.5rem" }}>
          <div>
            <h3 className="card-title">Per-Class Test Performance</h3>
            <p className="card-subtitle">Precision, Recall, and F1 metrics evaluated on 400 test images per category</p>
          </div>
          <span style={{ fontSize: "0.75rem", padding: "0.25rem 0.65rem", backgroundColor: "var(--surface-secondary)", border: "1px solid var(--border)", borderRadius: "var(--radius-full)", color: "var(--text-muted)" }}>
            Balanced 400 Scans / Class
          </span>
        </div>

        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Class Category</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Test Support</th>
                <th>Clinical / Diagnostic Observation</th>
              </tr>
            </thead>
            <tbody>
              {PER_CLASS_PERFORMANCE.map((row, idx) => (
                <tr key={idx}>
                  <td>
                    <strong style={{ color: "var(--text-primary)" }}>{row.className}</strong>
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)", fontWeight: "600" }}>{row.precision}</td>
                  <td style={{ fontFamily: "var(--font-mono)", fontWeight: "600" }}>{row.recall}</td>
                  <td style={{ fontFamily: "var(--font-mono)", fontWeight: "600" }}>{row.f1}</td>
                  <td>{row.correct} / {row.total}</td>
                  <td style={{ fontSize: "0.8rem", color: "var(--text-muted)", maxWidth: "340px" }}>{row.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Confusion Matrix Section */}
      <section className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem", flexWrap: "wrap", gap: "1rem" }}>
          <div>
            <h3 className="card-title">Empirical Confusion Matrix</h3>
            <p className="card-subtitle">
              Directional true class (rows) versus predicted class (columns) on 1,600 held-out test scans
            </p>
          </div>

          {/* Matrix Tab Switcher */}
          <div style={{ display: "flex", gap: "0.35rem", backgroundColor: "var(--surface-secondary)", padding: "0.25rem", borderRadius: "var(--radius-md)", border: "1px solid var(--border)" }}>
            <button
              className={`btn btn-sm ${activeMatrixTab === "raw" ? "btn-secondary" : ""}`}
              style={{ padding: "0.35rem 0.75rem", fontSize: "0.78rem" }}
              onClick={() => setActiveMatrixTab("raw")}
            >
              Raw Counts (N=1,600)
            </button>
            <button
              className={`btn btn-sm ${activeMatrixTab === "normalized" ? "btn-secondary" : ""}`}
              style={{ padding: "0.35rem 0.75rem", fontSize: "0.78rem" }}
              onClick={() => setActiveMatrixTab("normalized")}
            >
              Normalized (%)
            </button>
          </div>
        </div>

        <div
          className="matrix-container"
          onClick={() =>
            setModalImage({
              title: activeMatrixTab === "raw" ? "Confusion Matrix (Raw Counts)" : "Confusion Matrix (Normalized %)",
              src: activeMatrixTab === "raw" ? "/results/confusion_matrix.png" : "/results/confusion_matrix_normalized.png",
              caption: activeMatrixTab === "raw"
                ? "Total test predictions across 400 scans per class. Main diagonal represents 1,485 correct classifications."
                : "Row-normalized percentages. Pituitary achieved 100.0% accuracy; No Tumor achieved 99.25% recall.",
            })
          }
          title="Click to enlarge confusion matrix"
        >
          <img
            src={activeMatrixTab === "raw" ? "/results/confusion_matrix.png" : "/results/confusion_matrix_normalized.png"}
            alt="Confusion Matrix"
            loading="lazy"
          />
          <div className="gradcam-zoom-hint">Click to enlarge 🔍</div>
        </div>

        <div style={{ marginTop: "1rem", fontSize: "0.8rem", color: "var(--text-muted)", lineHeight: "1.5" }}>
          <strong>Matrix Interpretation:</strong> The diagonal entries represent correct predictions ($N = 1,485$). Off-diagonal values highlight specific failure modes: most prominently, 56 glioma images were classified as meningioma, and 25 glioma images were classified as healthy tissue.
        </div>
      </section>

      {/* Zoom Modal */}
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
            {modalImage.caption}
          </p>
        </Modal>
      )}
    </div>
  );
}
