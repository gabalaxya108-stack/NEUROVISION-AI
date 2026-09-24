import React from "react";
import { CLASS_METADATA } from "../utils/constants";

export default function PredictionCard({ prediction }) {
  if (!prediction) return null;

  const { predicted_class, confidence, probabilities } = prediction;
  const meta = CLASS_METADATA[predicted_class] || {
    name: predicted_class,
    badgeColor: "#2563eb",
    bgColor: "#eff6ff",
    borderColor: "#bfdbfe",
    description: "",
  };

  const confidencePct = (confidence * 100).toFixed(2);

  // Ordered classes
  const classOrder = ["glioma", "meningioma", "pituitary", "notumor"];

  return (
    <div className="prediction-card">
      <div className="prediction-header">
        <div>
          <span
            className="prediction-badge"
            style={{
              backgroundColor: meta.bgColor,
              color: meta.badgeColor,
              border: `1px solid ${meta.borderColor}`,
            }}
          >
            Predicted Class
          </span>
          <div className="prediction-winner">{meta.name}</div>
          {meta.description && (
            <p style={{ fontSize: "0.825rem", color: "var(--color-text-subtle)", marginTop: "0.25rem" }}>
              {meta.description}
            </p>
          )}
        </div>

        <div className="confidence-score-block">
          <div className="confidence-label">Model confidence score</div>
          <div className="confidence-number">{confidencePct}%</div>
        </div>
      </div>

      <div style={{ marginBottom: "0.75rem" }}>
        <h4 style={{ fontSize: "0.85rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--color-text-subtle)" }}>
          Class Scores Distribution
        </h4>
      </div>

      <div className="probabilities-list">
        {classOrder.map((clsKey) => {
          const clsMeta = CLASS_METADATA[clsKey];
          const rawScore = probabilities?.[clsKey] ?? 0;
          const scorePct = (rawScore * 100).toFixed(2);
          const isWinner = clsKey === predicted_class;

          return (
            <div key={clsKey} className="probability-row">
              <div className="prob-header">
                <span style={{ fontWeight: isWinner ? "700" : "500", color: isWinner ? "var(--color-text-main)" : "var(--color-text-muted)" }}>
                  {clsMeta?.name || clsKey} {isWinner && "✓"}
                </span>
                <span style={{ fontFamily: "var(--font-mono)", fontWeight: isWinner ? "700" : "500" }}>
                  {scorePct}%
                </span>
              </div>
              <div className="prob-track">
                <div
                  className="prob-fill"
                  style={{
                    width: `${scorePct}%`,
                    backgroundColor: clsMeta?.badgeColor || "#2563eb",
                    opacity: isWinner ? 1 : 0.65,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div
        style={{
          marginTop: "1.25rem",
          padding: "0.75rem 1rem",
          backgroundColor: "#f8fafc",
          borderRadius: "var(--radius-sm)",
          fontSize: "0.8rem",
          color: "var(--color-text-subtle)",
          lineHeight: 1.45,
        }}
      >
        <strong>Nomenclature Notice:</strong> "Model confidence score" refers strictly to the normalized softmax
        output of the convolutional classifier on this specific input. It represents statistical visual feature
        correlation and must not be interpreted as clinical diagnostic certainty or disease probability.
      </div>
    </div>
  );
}
