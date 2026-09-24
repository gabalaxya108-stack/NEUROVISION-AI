import React, { useState } from "react";
import StatCard from "../components/StatCard";
import Modal from "../components/Modal";
import { REPRESENTATIVE_ERRORS } from "../utils/constants";

export default function ErrorAnalysis() {
  const [modalItem, setModalItem] = useState(null);

  const errorStats = [
    { value: "115", label: "Incorrect Predictions", detail: "Total test classification errors" },
    { value: "7.19%", label: "Test Error Rate", detail: "115 errors out of 1,600 scans" },
    { value: "56", label: "Glioma → Meningioma", detail: "48.70% of all incorrect predictions" },
    { value: "56", label: "High-Confidence Errors", detail: "Errors with confidence ≥ 80%" },
    { value: "46", label: "Borderline Predictions", detail: "Scans with confidence < 60%" },
  ];

  return (
    <div className="container">
      {/* Title */}
      <div style={{ marginBottom: "1.75rem" }}>
        <h1 style={{ fontSize: "2rem", fontWeight: "800" }}>Comprehensive Error & Confidence Analysis</h1>
        <p style={{ color: "var(--text-muted)", fontSize: "0.95rem", marginTop: "0.25rem" }}>
          In-depth audit of test failure modes, confidence score distributions, and directional misclassification pairs.
        </p>
      </div>

      {/* 5 Key Error Stats */}
      <section className="error-stats-grid">
        {errorStats.map((item, idx) => (
          <StatCard
            key={idx}
            value={item.value}
            label={item.label}
            detail={item.detail}
            highlight={idx === 2}
          />
        ))}
      </section>

      {/* Directional Misclassification Observation Callout */}
      <div style={{ padding: "1.25rem 1.5rem", backgroundColor: "#f8fafc", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)", marginBottom: "2rem" }}>
        <div style={{ display: "flex", gap: "0.75rem", alignItems: "flex-start" }}>
          <span style={{ fontSize: "1.35rem", lineHeight: 1 }}>🔬</span>
          <div>
            <strong style={{ fontSize: "0.95rem", color: "var(--text-primary)" }}>
              Most Frequent Observed Misclassification Pair: Glioma &rarr; Meningioma (56 Scans)
            </strong>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: "0.35rem", lineHeight: "1.55" }}>
              48.70% of all test errors (56 out of 115) occurred when true glioma scans were predicted as meningioma. Morphological inspection indicates this arises when peripheral infiltrative gliomas contact dural surfaces, mimicking extra-axial meningioma intensity margins. In contrast, Meningioma &rarr; Glioma occurred only 18 times (15.65% of errors), demonstrating significant directional asymmetry.
            </p>
          </div>
        </div>
      </div>

      {/* Charts Grid */}
      <section className="charts-grid">
        <div className="chart-card">
          <h3 className="card-title">Misclassification Pairs Frequency</h3>
          <p className="card-subtitle">Directional error breakdown across all actual &rarr; predicted combinations</p>
          <div
            className="chart-wrapper"
            onClick={() =>
              setModalItem({
                title: "Directional Misclassification Pairs (N=115)",
                imageSrc: "/results/misclassification_pairs.png",
                caption: "Dominated by glioma → meningioma (56 cases) and glioma → notumor (25 cases).",
              })
            }
          >
            <img src="/results/misclassification_pairs.png" alt="Misclassification Pairs Chart" loading="lazy" />
            <div className="gradcam-zoom-hint">Click to enlarge 🔍</div>
          </div>
        </div>

        <div className="chart-card">
          <h3 className="card-title">Confidence Distribution Calibration</h3>
          <p className="card-subtitle">Correct (Mean: 97.29%) vs. Incorrect (Mean: 75.91%) model confidence scores</p>
          <div
            className="chart-wrapper"
            onClick={() =>
              setModalItem({
                title: "Model Confidence Score Distributions",
                imageSrc: "/results/confidence_correct_vs_incorrect.png",
                caption: "Correct predictions cluster strongly near 1.0; incorrect predictions exhibit broader uncertainty spread.",
              })
            }
          >
            <img src="/results/confidence_correct_vs_incorrect.png" alt="Confidence Distribution Comparison" loading="lazy" />
            <div className="gradcam-zoom-hint">Click to enlarge 🔍</div>
          </div>
        </div>
      </section>

      {/* Second Row: Per-Class Confidence Boxplot */}
      <section className="card" style={{ marginBottom: "2rem" }}>
        <h3 className="card-title">Confidence Score Calibration Across Classes</h3>
        <p className="card-subtitle">Distribution of model confidence scores for each target class on test scans</p>

        <div
          className="chart-wrapper"
          style={{ marginTop: "1.25rem" }}
          onClick={() =>
            setModalItem({
              title: "Confidence Distributions by Class",
              imageSrc: "/results/confidence_by_class.png",
              caption: "Pituitary and No Tumor exhibit near-perfect tight distributions (median > 0.99). Glioma shows the widest spread due to morphological variability.",
            })
          }
        >
          <img
            src="/results/confidence_by_class.png"
            alt="Confidence by Class Boxplot"
            style={{ maxHeight: "380px" }}
            loading="lazy"
          />
          <div className="gradcam-zoom-hint">Click to enlarge 🔍</div>
        </div>
      </section>

      {/* Representative High-Confidence Error Case Studies */}
      <section className="card">
        <h3 className="card-title">Representative High-Confidence Error Case Studies</h3>
        <p className="card-subtitle">
          Audited failure cases evaluated with Grad-CAM explainability to understand convolutional activation cues
        </p>

        <div className="error-cases-grid">
          {REPRESENTATIVE_ERRORS.map((item, idx) => (
            <div key={idx} className="error-case-card">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: "0.8rem", fontWeight: "700", color: "var(--text-primary)" }}>
                  Case ID: <span className="font-mono">{item.id}</span>
                </span>
                <span style={{ fontSize: "0.72rem", padding: "0.2rem 0.55rem", backgroundColor: "#fee2e2", color: "#991b1b", borderRadius: "var(--radius-full)", fontWeight: "700" }}>
                  Confidence: {item.confidence}
                </span>
              </div>

              <div style={{ display: "flex", gap: "1rem", fontSize: "0.8rem" }}>
                <div>
                  <span style={{ color: "var(--text-muted)" }}>Actual: </span>
                  <strong>{item.actual}</strong>
                </div>
                <div>
                  <span style={{ color: "var(--text-muted)" }}>Predicted: </span>
                  <strong style={{ color: "var(--error)" }}>{item.predicted}</strong>
                </div>
              </div>

              <div
                style={{
                  height: "180px",
                  backgroundColor: "#0b1120",
                  borderRadius: "var(--radius-sm)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  cursor: "pointer",
                  position: "relative",
                  overflow: "hidden",
                }}
                onClick={() =>
                  setModalItem({
                    title: `Case ${item.id}: Grad-CAM Error Attribution`,
                    imageSrc: item.gradcamPath,
                    caption: `Actual: ${item.actual} | Predicted: ${item.predicted} (${item.confidence} confidence score). Note: Highlighted regions indicate model activation patterns, not confirmed anatomical lesions.`,
                  })
                }
              >
                <img
                  src={item.gradcamPath}
                  alt={`Grad-CAM for ${item.id}`}
                  style={{ maxHeight: "100%", maxWidth: "100%", objectFit: "contain" }}
                  loading="lazy"
                />
                <div className="gradcam-zoom-hint">🔍 View Grad-CAM</div>
              </div>

              <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", lineHeight: "1.45" }}>
                <strong>Attribution Insight:</strong> {item.analysis}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Zoom Modal */}
      {modalItem && (
        <Modal
          isOpen={Boolean(modalItem)}
          onClose={() => setModalItem(null)}
          title={modalItem.title}
        >
          <img
            src={modalItem.imageSrc}
            alt={modalItem.title}
            style={{ maxWidth: "100%", maxHeight: "56vh", objectFit: "contain" }}
          />
          <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginTop: "1rem", textAlign: "center" }}>
            {modalItem.caption}
          </p>
        </Modal>
      )}
    </div>
  );
}
