import React, { useState } from "react";
import { DISCLAIMER_TEXT } from "../utils/constants";

export default function Methodology() {
  const [expandedStep, setExpandedStep] = useState(null);

  const pipelineSteps = [
    {
      num: 1,
      title: "Dataset Acquisition & MD5 Verification",
      badge: "7,200 Scans",
      summary: "Balanced multi-class brain MRI scans across 4 categories.",
      details: "7,200 total brain MRI images partitioned into 5,600 training scans and 1,600 test scans. Verified complete decodability, valid file extensions (.jpg), uniform bit depth, and strictly zero cross-split exact duplicates via MD5 image hashing.",
      specs: [
        { key: "Total Images", val: "7,200" },
        { key: "Classes", val: "4 (glioma, meningioma, pituitary, notumor)" },
        { key: "Class Balance", val: "1,800 images / class overall" },
        { key: "Integrity", val: "Zero cross-split duplicate hashes" },
      ],
    },
    {
      num: 2,
      title: "Aspect-Preserving Letterbox Preprocessing",
      badge: "224×224×3 RGB",
      summary: "Deterministic letterbox resizing avoiding anatomical deformation.",
      details: "Raw MRI scans have varying resolutions (from 256×256 to 800×800). Direct isotropic resizing deforms delicate cranial morphology. We implement LetterboxResize: the image aspect ratio is strictly preserved, scaled so the largest dimension equals 224px, and centered with black zero-padding to reach 224×224×3 RGB.",
      specs: [
        { key: "Target Dimensions", val: "224 × 224 × 3" },
        { key: "Channels", val: "3-channel RGB" },
        { key: "Aspect Ratio", val: "Preserved (Zero padding)" },
        { key: "Interpolation", val: "Bilinear (PIL.Image.BILINEAR)" },
      ],
    },
    {
      num: 3,
      title: "Reproducible Train / Validation Split",
      badge: "80 / 20 Partition",
      summary: "Partitioned with fixed seed 42 without touching test data.",
      details: "The 5,600 training images were partitioned into 80% Training (4,480 images, 1,120/class) and 20% Validation (1,120 images, 280/class). The 1,600 test scans in archive/Testing were locked and held completely untouched until final benchmark evaluation.",
      specs: [
        { key: "Training Subset", val: "4,480 scans (80%)" },
        { key: "Validation Subset", val: "1,120 scans (20%)" },
        { key: "Held-out Test Subset", val: "1,600 scans (Untouched)" },
        { key: "Random Seed", val: "Fixed 42 (Reproducible)" },
      ],
    },
    {
      num: 4,
      title: "Two-Phase Transfer Learning (EfficientNetB0)",
      badge: "4.05M Params",
      summary: "ImageNet pre-trained feature extraction followed by top-layer fine-tuning.",
      details: "Backbone initialized with ImageNet pre-trained weights. Phase 1 (Epochs 1–10): Backbone feature extractor frozen, trained 4-class classification head with Adam optimizer (lr=1e-3). Phase 2 (Epochs 11–20): Top 25 convolutional layers unfrozen, trained with reduced learning rate (lr=1e-4) for domain adaptation.",
      specs: [
        { key: "Backbone", val: "EfficientNetB0 (ImageNet)" },
        { key: "Phase 1", val: "Head training (10 epochs, lr=1e-3)" },
        { key: "Phase 2", val: "Fine-tune top 25 layers (10 epochs, lr=1e-4)" },
        { key: "Total Parameters", val: "4,054,772" },
      ],
    },
    {
      num: 5,
      title: "Validation Checkpoint Selection",
      badge: "96.25% Val Acc",
      summary: "Optimal model weights saved based on validation loss and accuracy.",
      details: "Validation loss and categorical accuracy monitored after each epoch. Best model checkpoint saved automatically at Epoch 20: validation loss reached 0.1059 and validation accuracy achieved 96.25%, demonstrating strong generalization without overfitting.",
      specs: [
        { key: "Best Checkpoint", val: "Epoch 20" },
        { key: "Validation Accuracy", val: "96.25%" },
        { key: "Validation Loss", val: "0.1059" },
        { key: "File Path", val: "models/best_brain_tumor_model.keras" },
      ],
    },
    {
      num: 6,
      title: "Held-Out Test Benchmark Evaluation",
      badge: "92.81% Test Acc",
      summary: "Empirical evaluation on untouched 1,600-image test set.",
      details: "The best model checkpoint evaluated on the 1,600 held-out test scans (400 per class). Achieved 92.81% overall accuracy (1,485 / 1,600 correct), 93.13% macro precision, 92.81% macro recall, and 92.63% macro F1-score without test-set leakage.",
      specs: [
        { key: "Untouched Test Scans", val: "1,600 (400 / class)" },
        { key: "Overall Accuracy", val: "92.81% (1,485 / 1,600)" },
        { key: "Macro F1-Score", val: "92.63%" },
        { key: "Macro Precision", val: "93.13%" },
      ],
    },
    {
      num: 7,
      title: "Mandatory Brain MRI Input Guardrail",
      badge: "Modality Gate",
      summary: "Pre-inference validation layer preventing arbitrary non-MRI analysis.",
      details: "A dedicated module (src/input_guard.py) validates image inputs before the tumor classifier. Enforces 5 deterministic stages: decoding integrity, chromatic saturation in tissue pixels, perimeter border brightness, cranial mass morphology/centroid, and rectilinear edge filtering. Non-MRI images and corrupted files are safely rejected server-side.",
      specs: [
        { key: "Module", val: "src/input_guard.py" },
        { key: "Decision Gate", val: "VALID_MRI | INVALID_NON_MRI | UNCERTAIN" },
        { key: "Classifier Intercept", val: "0 calls on rejected / uncertain inputs" },
        { key: "Enforcement", val: "Server-side (/predict & /explain)" },
      ],
    },
    {
      num: 8,
      title: "Explainable AI (Grad-CAM Attribution)",
      badge: "top_activation",
      summary: "Gradient-weighted class activation mapping targeting final conv layer.",
      details: "Grad-CAM computes the gradients of the top predicted class score with respect to the final convolutional feature maps ('top_activation', 7×7×1280). Global average pooling generates channel importance weights, combined via ReLU and JET colormap, then mapped back via inverse un-letterboxing to overlay on the original scan.",
      specs: [
        { key: "Target Layer", val: "top_activation (7×7×1280)" },
        { key: "Colormap", val: "JET (40% blend with 60% MRI)" },
        { key: "Un-letterboxing", val: "Exact bounding box coordinate mapping" },
        { key: "Implementation", val: "src/explain.py (Headless Agg backend)" },
      ],
    },
    {
      num: 9,
      title: "Inference Backend & Research Platform",
      badge: "FastAPI + Vite",
      summary: "Production REST API with in-memory processing and accessible SPA.",
      details: "FastAPI inference backend with resident model in memory, serving /health, /model-info, /predict, and /explain. Uploaded scans processed strictly in-memory via io.BytesIO and never permanently stored. React/Vite research interface provides interactive inspection and zero dataset exposure via SafeStaticFiles.",
      specs: [
        { key: "API Framework", val: "FastAPI 0.115 + Uvicorn" },
        { key: "Frontend", val: "React 19 + Vite 8 SPA" },
        { key: "Dataset Isolation", val: "SafeStaticFiles blocks CSVs & archive" },
        { key: "Privacy", val: "Zero disk persistence for user uploads" },
      ],
    },
  ];

  return (
    <div className="container">
      {/* Title */}
      <div style={{ marginBottom: "1.75rem" }}>
        <h1 style={{ fontSize: "2rem", fontWeight: "800" }}>System Methodology & Research Pipeline</h1>
        <p style={{ color: "var(--text-muted)", fontSize: "0.95rem", marginTop: "0.25rem" }}>
          End-to-end deep learning engineering pipeline from raw DICOM/JPEG MRI slices to explainable web deployment.
        </p>
      </div>

      {/* Dataset & Architecture Specifications Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "1.5rem", marginBottom: "2rem" }}>
        <div className="card">
          <h3 className="card-title">Dataset Partition Distribution</h3>
          <p className="card-subtitle">7,200 Total Brain MRI Scans (1,800 per class)</p>

          <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem", marginTop: "1rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", paddingBottom: "0.5rem", borderBottom: "1px solid var(--border)" }}>
              <span style={{ fontSize: "0.85rem" }}>Training Split (80% of Training):</span>
              <strong style={{ fontSize: "0.85rem", fontFamily: "var(--font-mono)" }}>4,480 scans (1,120 / class)</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", paddingBottom: "0.5rem", borderBottom: "1px solid var(--border)" }}>
              <span style={{ fontSize: "0.85rem" }}>Validation Split (20% of Training):</span>
              <strong style={{ fontSize: "0.85rem", fontFamily: "var(--font-mono)" }}>1,120 scans (280 / class)</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", paddingBottom: "0.5rem", borderBottom: "1px solid var(--border)" }}>
              <span style={{ fontSize: "0.85rem" }}>Held-Out Test Set (Untouched):</span>
              <strong style={{ fontSize: "0.85rem", fontFamily: "var(--font-mono)", color: "var(--primary-accent)" }}>1,600 scans (400 / class)</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ fontSize: "0.85rem" }}>Total Scans:</span>
              <strong style={{ fontSize: "0.85rem", fontFamily: "var(--font-mono)" }}>7,200 scans (1,800 / class)</strong>
            </div>
          </div>
        </div>

        <div className="card">
          <h3 className="card-title">Transfer Learning Hyperparameters</h3>
          <p className="card-subtitle">Two-Stage Fine-Tuning Schedule</p>

          <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem", marginTop: "1rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", paddingBottom: "0.5rem", borderBottom: "1px solid var(--border)" }}>
              <span style={{ fontSize: "0.85rem" }}>Phase 1 (Epochs 1–10):</span>
              <span style={{ fontSize: "0.85rem", fontFamily: "var(--font-mono)" }}>Frozen backbone, lr = 1e-3</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", paddingBottom: "0.5rem", borderBottom: "1px solid var(--border)" }}>
              <span style={{ fontSize: "0.85rem" }}>Phase 2 (Epochs 11–20):</span>
              <span style={{ fontSize: "0.85rem", fontFamily: "var(--font-mono)" }}>Top 25 layers unfrozen, lr = 1e-4</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", paddingBottom: "0.5rem", borderBottom: "1px solid var(--border)" }}>
              <span style={{ fontSize: "0.85rem" }}>Batch Size / Optimizer:</span>
              <span style={{ fontSize: "0.85rem", fontFamily: "var(--font-mono)" }}>32 / Adam (beta1=0.9, beta2=0.999)</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ fontSize: "0.85rem" }}>Loss Function:</span>
              <span style={{ fontSize: "0.85rem", fontFamily: "var(--font-mono)" }}>Categorical Crossentropy</span>
            </div>
          </div>
        </div>
      </div>

      {/* Interactive Step-by-Step Pipeline */}
      <section className="card" style={{ marginBottom: "2rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.5rem" }}>
          <div>
            <h3 className="card-title">Interactive Pipeline Architecture</h3>
            <p className="card-subtitle">Click any stage to view in-depth engineering specifications & parameters</p>
          </div>
          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
            9 Sequential Stages
          </span>
        </div>

        <div className="pipeline-steps-list">
          {pipelineSteps.map((step) => {
            const isExpanded = expandedStep === step.num;
            return (
              <div
                key={step.num}
                className="pipeline-step-card"
                onClick={() => setExpandedStep(isExpanded ? null : step.num)}
              >
                <div className="pipeline-step-num">{step.num}</div>
                <div className="pipeline-step-content">
                  <div className="pipeline-step-header">
                    <span className="pipeline-step-title">{step.title}</span>
                    <span className="pipeline-step-badge">{step.badge}</span>
                  </div>

                  <p className="pipeline-step-desc">{step.summary}</p>

                  {/* Expandable Technical Details */}
                  {isExpanded && (
                    <div style={{ marginTop: "1rem", paddingTop: "0.85rem", borderTop: "1px solid var(--border)", animation: "fadeIn 150ms ease-out" }}>
                      <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: "1.6", marginBottom: "0.85rem" }}>
                        {step.details}
                      </p>

                      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.5rem", backgroundColor: "var(--surface-secondary)", padding: "0.75rem 1rem", borderRadius: "var(--radius-sm)" }}>
                        {step.specs.map((s, idx) => (
                          <div key={idx} style={{ fontSize: "0.78rem" }}>
                            <span style={{ color: "var(--text-muted)" }}>{s.key}: </span>
                            <strong style={{ color: "var(--text-primary)" }}>{s.val}</strong>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                <div style={{ fontSize: "1.1rem", color: "var(--text-subtle)", padding: "0.25rem" }}>
                  {isExpanded ? "▲" : "▼"}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Research Disclaimer */}
      <section style={{ padding: "1.25rem 1.5rem", backgroundColor: "#fffbeb", border: "1px solid #fde68a", borderRadius: "var(--radius-lg)" }}>
        <strong style={{ fontSize: "0.9rem", color: "#92400e" }}>Academic Research Notice:</strong>
        <p style={{ fontSize: "0.85rem", color: "#92400e", marginTop: "0.25rem", lineHeight: "1.5" }}>
          {DISCLAIMER_TEXT} The methodology presented represents a controlled computer-vision benchmark and should not be used as clinical diagnostic software.
        </p>
      </section>
    </div>
  );
}
