import React from "react";
import StatCard from "../components/StatCard";
import { PROJECT_STATS, CLASS_METADATA, DISCLAIMER_TEXT } from "../utils/constants";

export default function Dashboard({ setActivePage, onOpenAbout }) {
  return (
    <div className="container">
      {/* Hero Section */}
      <section className="dashboard-hero">
        <div className="hero-tag">
          <span style={{ fontSize: "0.9rem" }}>🔬</span> Academic Research & Evaluation Platform
        </div>

        <h1 className="hero-title">
          Brain MRI Classification
        </h1>

        <p className="hero-subtitle">
          Deep learning research platform for four-class MRI image classification. Powered by an EfficientNetB0
          backbone with visual Grad-CAM explainability, rigorous error calibration, and an integrated input guardrail.
        </p>

        <div className="hero-actions">
          <button className="btn btn-primary" onClick={() => setActivePage("analyze")}>
            Analyze MRI &rarr;
          </button>
          <button className="btn btn-secondary" onClick={() => setActivePage("performance")}>
            Explore Model Performance
          </button>
          {onOpenAbout && (
            <button className="btn btn-secondary" onClick={onOpenAbout}>
              About Platform
            </button>
          )}
        </div>
      </section>

      {/* 4 Core Project Statistics */}
      <section>
        <div className="stats-grid">
          {PROJECT_STATS.map((stat, idx) => (
            <StatCard
              key={idx}
              value={stat.value}
              label={stat.label}
              detail={stat.detail}
              highlight={idx === 2}
            />
          ))}
        </div>
      </section>

      {/* Four Target Classes Overview */}
      <section style={{ marginTop: "1rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: "0.5rem", marginBottom: "1rem" }}>
          <div>
            <h2 style={{ fontSize: "1.4rem", fontWeight: "800" }}>Four-Class Categorization Schema</h2>
            <p style={{ color: "var(--text-muted)", fontSize: "0.875rem", marginTop: "0.2rem" }}>
              Target disease categories and normal neuroanatomical controls evaluated in 224&times;224 space
            </p>
          </div>
          <button
            className="btn btn-secondary btn-sm"
            onClick={() => setActivePage("performance")}
          >
            View Benchmark Breakdown &rarr;
          </button>
        </div>

        <div className="classes-grid">
          {Object.entries(CLASS_METADATA).map(([key, item]) => (
            <div key={key} className="class-overview-card">
              <div>
                <div className="class-card-top">
                  <span
                    className="class-card-badge"
                    style={{
                      backgroundColor: item.bgColor,
                      color: item.badgeColor,
                      border: `1px solid ${item.borderColor}`,
                    }}
                  >
                    Class {item.id}
                  </span>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-subtle)", fontWeight: "600" }}>
                    400 test scans
                  </span>
                </div>
                <h3 className="class-card-name">{item.name}</h3>
                <p className="class-card-desc">{item.description}</p>
              </div>

              <div className="class-card-meta">
                <span>Hold-out Support: 400</span>
                <span style={{ fontWeight: "600", color: item.badgeColor }}>
                  {key === "pituitary" ? "100.0% Recall" : key === "notumor" ? "99.25% Recall" : key === "meningioma" ? "94.00% Recall" : "97.20% Precision"}
                </span>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Model Architecture & Guardrail Summary */}
      <section className="architecture-card">
        <div>
          <div style={{ display: "inline-block", fontSize: "0.72rem", fontWeight: "800", textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--accent)", marginBottom: "0.5rem" }}>
            Neural Architecture
          </div>
          <h3 style={{ fontSize: "1.35rem", fontWeight: "800", marginBottom: "0.6rem" }}>
            EfficientNetB0 Backbone + Input Guardrail
          </h3>
          <p style={{ fontSize: "0.875rem", color: "var(--text-muted)", lineHeight: "1.6" }}>
            Trained in two reproducible phases: frozen feature extraction followed by fine-tuning top convolutional layers.
            Every input is guarded by a multi-stage modality verification layer to prevent arbitrary non-MRI images from reaching the classifier.
          </p>

          <div style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap", marginTop: "1rem" }}>
            <span style={{ fontSize: "0.75rem", padding: "0.25rem 0.6rem", backgroundColor: "var(--surface-secondary)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", fontFamily: "var(--font-mono)" }}>
              ImageNet Pre-trained
            </span>
            <span style={{ fontSize: "0.75rem", padding: "0.25rem 0.6rem", backgroundColor: "var(--surface-secondary)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", fontFamily: "var(--font-mono)" }}>
              224&times;224 Letterbox
            </span>
            <span style={{ fontSize: "0.75rem", padding: "0.25rem 0.6rem", backgroundColor: "var(--surface-secondary)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", fontFamily: "var(--font-mono)" }}>
              4.05M Parameters
            </span>
            <span style={{ fontSize: "0.75rem", padding: "0.25rem 0.6rem", backgroundColor: "#f0fdf4", color: "#16a34a", border: "1px solid #bbf7d0", borderRadius: "var(--radius-sm)", fontWeight: "600" }}>
              Server Guardrail Active
            </span>
          </div>
        </div>

        <div style={{ backgroundColor: "var(--surface-secondary)", padding: "1.25rem", borderRadius: "var(--radius-md)", border: "1px solid var(--border)" }}>
          <h4 style={{ fontSize: "0.95rem", fontWeight: "700", marginBottom: "0.5rem" }}>
            Verified Pipeline Sequence
          </h4>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", fontSize: "0.8rem", color: "var(--text-secondary)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span style={{ width: "18px", height: "18px", borderRadius: "50%", backgroundColor: "var(--primary)", color: "white", display: "inline-flex", alignItems: "center", justifyContent: "center", fontSize: "0.65rem", fontWeight: "700" }}>1</span>
              <span><strong>Input Guardrail:</strong> Decodes & validates grayscale cranial morphology</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span style={{ width: "18px", height: "18px", borderRadius: "50%", backgroundColor: "var(--primary)", color: "white", display: "inline-flex", alignItems: "center", justifyContent: "center", fontSize: "0.65rem", fontWeight: "700" }}>2</span>
              <span><strong>Letterbox Preprocessing:</strong> Preserves anatomical aspect ratio (224&times;224)</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span style={{ width: "18px", height: "18px", borderRadius: "50%", backgroundColor: "var(--primary)", color: "white", display: "inline-flex", alignItems: "center", justifyContent: "center", fontSize: "0.65rem", fontWeight: "700" }}>3</span>
              <span><strong>EfficientNetB0 Classifier:</strong> Outputs calibrated 4-class probabilities</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span style={{ width: "18px", height: "18px", borderRadius: "50%", backgroundColor: "var(--primary)", color: "white", display: "inline-flex", alignItems: "center", justifyContent: "center", fontSize: "0.65rem", fontWeight: "700" }}>4</span>
              <span><strong>Grad-CAM Attribution:</strong> Visualizes spatial feature activation</span>
            </div>
          </div>
        </div>
      </section>

      {/* Prominent Educational Notice */}
      <section style={{ marginTop: "2rem" }}>
        <div style={{ padding: "1.25rem", backgroundColor: "#fffbeb", border: "1px solid #fde68a", borderRadius: "var(--radius-lg)", display: "flex", gap: "0.85rem", alignItems: "flex-start" }}>
          <span style={{ fontSize: "1.5rem", lineHeight: 1 }}>⚖️</span>
          <div>
            <strong style={{ fontSize: "0.9rem", color: "#92400e" }}>Academic Research Notice:</strong>
            <p style={{ fontSize: "0.85rem", color: "#92400e", marginTop: "0.25rem", lineHeight: "1.5" }}>
              {DISCLAIMER_TEXT} The system is built as a reproducible machine learning benchmark and must not be used for diagnosis or clinical patient management.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
