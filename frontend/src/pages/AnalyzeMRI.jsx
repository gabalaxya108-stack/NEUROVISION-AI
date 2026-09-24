import React, { useState, useRef } from "react";
import PredictionCard from "../components/PredictionCard";
import GradcamViewer from "../components/GradcamViewer";
import Modal from "../components/Modal";
import { api, ApiError } from "../services/api";
import { SAMPLE_IMAGES, GUARDRAIL_TEST_SAMPLES } from "../utils/constants";

export default function AnalyzeMRI() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreviewSrc, setImagePreviewSrc] = useState(null);
  const [imageDimensions, setImageDimensions] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  // Workflow states: 'initial' | 'ready' | 'analyzing' | 'success' | 'rejected' | 'uncertain' | 'error'
  const [status, setStatus] = useState("initial");
  const [errorMessage, setErrorMessage] = useState(null);
  const [guardrailRejection, setGuardrailRejection] = useState(null);

  const [prediction, setPrediction] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [isExplaining, setIsExplaining] = useState(false);
  const [modalImage, setModalImage] = useState(null);

  const fileInputRef = useRef(null);

  // --- File Selection & Validation ---
  const handleFileProcess = (file) => {
    if (!file) return;

    // Validate MIME / extension
    const validTypes = ["image/jpeg", "image/jpg", "image/png"];
    const ext = file.name.split(".").pop().toLowerCase();
    if (!validTypes.includes(file.type) && !["jpg", "jpeg", "png"].includes(ext)) {
      setStatus("error");
      setErrorMessage("Unsupported file type. Please upload a valid JPG or PNG image.");
      setSelectedFile(null);
      setImagePreviewSrc(null);
      return;
    }

    // Validate size (10MB)
    if (file.size > 10 * 1024 * 1024) {
      setStatus("error");
      setErrorMessage("File size exceeds 10MB limit. Please upload an image under 10MB.");
      return;
    }

    setSelectedFile(file);
    setStatus("ready");
    setErrorMessage(null);
    setGuardrailRejection(null);
    setPrediction(null);
    setExplanation(null);

    // Read preview & dimensions
    const reader = new FileReader();
    reader.onload = (e) => {
      const src = e.target.result;
      setImagePreviewSrc(src);

      const img = new Image();
      img.onload = () => {
        setImageDimensions({ width: img.naturalWidth, height: img.naturalHeight });
      };
      img.src = src;
    };
    reader.readAsDataURL(file);
  };

  // Drag and Drop handlers
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileProcess(e.dataTransfer.files[0]);
    }
  };

  // Load demonstration sample image
  const handleSelectSample = async (samplePath) => {
    try {
      setStatus("ready");
      setErrorMessage(null);
      setGuardrailRejection(null);
      setPrediction(null);
      setExplanation(null);

      const response = await fetch(samplePath);
      if (!response.ok) throw new Error("Could not load sample file.");
      const blob = await response.blob();
      const filename = samplePath.split("/").pop();
      const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
      handleFileProcess(file);
    } catch {
      setStatus("error");
      setErrorMessage("Unable to load demonstration sample image.");
    }
  };

  // Reset Flow (Section 7: Clear all temporary state)
  const handleReset = () => {
    setSelectedFile(null);
    setImagePreviewSrc(null);
    setImageDimensions(null);
    setPrediction(null);
    setExplanation(null);
    setGuardrailRejection(null);
    setStatus("initial");
    setErrorMessage(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // --- Run Inference & Guardrail Verification ---
  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setStatus("analyzing");
    setErrorMessage(null);
    setGuardrailRejection(null);
    setPrediction(null);
    setExplanation(null);

    try {
      // 1. Run /predict (server-side input guardrail evaluates image first)
      const predResult = await api.predictImage(selectedFile, selectedFile.name);

      // --- GUARDRAIL INTERCEPT CHECK ---
      if (!predResult.accepted) {
        // Input was rejected or uncertain -> STOP! Do not display tumor predictions or Grad-CAM
        if (predResult.input_type === "non_brain_mri") {
          setStatus("rejected");
        } else {
          setStatus("uncertain");
        }
        setGuardrailRejection({
          inputType: predResult.input_type,
          reason: predResult.reason,
          message: predResult.message || "This image does not appear to be a brain MRI.",
        });
        return;
      }

      // Valid Brain MRI verified
      setPrediction(predResult);
      setStatus("success");

      // 2. Request Grad-CAM /explain in parallel
      setIsExplaining(true);
      try {
        const explainResult = await api.explainImage(selectedFile, selectedFile.name);
        if (explainResult.accepted) {
          setExplanation(explainResult);
        }
      } catch (expErr) {
        console.warn("Grad-CAM generation notice:", expErr);
      } finally {
        setIsExplaining(false);
      }
    } catch (err) {
      setStatus("error");
      if (err instanceof ApiError && err.status === 503) {
        setErrorMessage("Unable to connect to the analysis service. Please try again.");
      } else {
        setErrorMessage(err.message || "An unexpected error occurred while analyzing the image.");
      }
    }
  };

  return (
    <div className="container">
      {/* Page Title */}
      <div style={{ marginBottom: "1.75rem" }}>
        <h1 style={{ fontSize: "2rem", fontWeight: "800" }}>Analyze Brain MRI Scan</h1>
        <p style={{ color: "var(--text-muted)", fontSize: "0.95rem", marginTop: "0.25rem" }}>
          Protected by a mandatory multi-stage Brain MRI input guardrail before convolutional classification.
        </p>
      </div>

      {status === "success" && prediction ? (
        /* --- HORIZONTAL RESEARCH DASHBOARD (Analyzed Success) --- */
        <div className="analyze-success-layout">
          {/* 1. Full-Width Verification Status Bar */}
          <div className="verified-action-bar">
            <div className="verified-status-info">
              <span className="verified-badge-icon">✓</span>
              <div>
                <div className="verified-status-title">Brain MRI Verified & Classified Successfully</div>
                <div className="verified-status-meta">
                  Modality Guardrail: Passed • Preprocessed: 224×224 RGB (Deterministic Letterbox) • Backbone: EfficientNetB0
                </div>
              </div>
            </div>

            <div className="verified-actions">
              <button className="btn btn-secondary btn-sm" onClick={handleReset}>
                ↺ Analyze Another Scan
              </button>
            </div>
          </div>

          {/* 2. Horizontal Analysis Row: Left Scan Card + Right Prediction Card */}
          <div className="analyzed-summary-grid">
            {/* Left: Input Scan Card */}
            <div className="card analyzed-scan-card">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
                <h4 style={{ fontSize: "0.95rem", fontWeight: "700" }}>Input MRI Scan</h4>
                <span className="badge badge-live">Verified MRI</span>
              </div>

              <div
                className="analyzed-scan-preview"
                onClick={() =>
                  setModalImage({
                    title: `Input Brain MRI: ${selectedFile?.name || "Scan"}`,
                    src: imagePreviewSrc,
                    desc: `${imageDimensions ? `${imageDimensions.width}×${imageDimensions.height} px` : "224×224 px"} • RGB 3-Channel Deterministic Letterbox Input`,
                  })
                }
                title="Click to enlarge input scan"
              >
                <img src={imagePreviewSrc} alt="Input Brain MRI Scan" />
                <div className="gradcam-zoom-hint">🔍 Enlarge</div>
              </div>

              <div className="analyzed-scan-details">
                <div className="scan-detail-row">
                  <span className="detail-key">Filename:</span>
                  <span className="detail-val" title={selectedFile?.name}>{selectedFile?.name}</span>
                </div>
                <div className="scan-detail-row">
                  <span className="detail-key">Dimensions:</span>
                  <span className="detail-val">{imageDimensions ? `${imageDimensions.width}×${imageDimensions.height} px` : "224×224 px"}</span>
                </div>
                <div className="scan-detail-row">
                  <span className="detail-key">Inference Tensor:</span>
                  <span className="detail-val">1×224×224×3 [0, 255]</span>
                </div>
                <div className="scan-detail-row">
                  <span className="detail-key">Memory Buffer:</span>
                  <span className="detail-val">Ephemeral Resident</span>
                </div>
              </div>

              {/* Quick Academic Test Samples Tray */}
              <div style={{ marginTop: "1rem", paddingTop: "0.85rem", borderTop: "1px solid var(--border)" }}>
                <div style={{ fontSize: "0.72rem", fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", marginBottom: "0.45rem" }}>
                  Quick Test Academic Samples:
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.4rem" }}>
                  {SAMPLE_IMAGES.map((sample) => (
                    <button
                      key={sample.id}
                      className="sample-pill-btn"
                      style={{ fontSize: "0.72rem", padding: "0.35rem 0.5rem", justifyContent: "center" }}
                      onClick={() => handleSelectSample(sample.path)}
                      disabled={status === "analyzing"}
                    >
                      🔬 {sample.expectedClass}
                    </button>
                  ))}
                </div>
              </div>

              {/* Quick Safety Guardrail Rejection Tray */}
              <div style={{ marginTop: "0.75rem", paddingTop: "0.75rem", borderTop: "1px solid var(--border)" }}>
                <div style={{ fontSize: "0.72rem", fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.05em", color: "#c05621", marginBottom: "0.45rem" }}>
                  🛡️ Test Guardrail Intercept:
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.4rem" }}>
                  {GUARDRAIL_TEST_SAMPLES.map((sample) => (
                    <button
                      key={sample.id}
                      className="sample-pill-btn"
                      style={{ fontSize: "0.72rem", padding: "0.35rem 0.5rem", justifyContent: "center", borderColor: "#fbd38d", color: "#9c4221", backgroundColor: "#fffaf0" }}
                      onClick={() => handleSelectSample(sample.path)}
                      disabled={status === "analyzing"}
                    >
                      ⚠️ {sample.title}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Right: Full Prediction & Multi-Class Distribution Card */}
            <div style={{ flex: 1, minWidth: 0 }}>
              <PredictionCard prediction={prediction} />
            </div>
          </div>

          {/* 3. Full-Width Horizontal Grad-CAM Explainability Section */}
          <div style={{ width: "100%", marginTop: "0.5rem" }}>
            <GradcamViewer
              originalImageSrc={imagePreviewSrc}
              explanation={explanation}
              isLoading={isExplaining}
            />
          </div>
        </div>
      ) : (
        /* --- UPLOAD & NON-SUCCESS WORKFLOW LAYOUT --- */
        <div className="analyze-layout">
          {/* Left Column: Upload & Input */}
          <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
            <div className="card">
              <h3 className="card-title" style={{ marginBottom: "0.75rem" }}>
                1. MRI Scan Input
              </h3>

              {/* Dropzone Area when no file selected */}
              {!selectedFile ? (
                <div
                  className={`dropzone-container ${dragActive ? "drag-active" : ""}`}
                  onDragEnter={handleDrag}
                  onDragOver={handleDrag}
                  onDragLeave={handleDrag}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <div className="dropzone-icon">
                    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                      <polyline points="17 8 12 3 7 8" />
                      <line x1="12" y1="3" x2="12" y2="15" />
                    </svg>
                  </div>

                  <div>
                    <div className="dropzone-title">Upload Brain MRI</div>
                    <div className="dropzone-subtitle" style={{ marginTop: "0.25rem" }}>
                      Drag and drop an MRI image here, or browse your files.
                    </div>
                  </div>

                  <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap", justifyContent: "center" }}>
                    <span style={{ fontSize: "0.72rem", padding: "0.2rem 0.5rem", backgroundColor: "var(--surface-secondary)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", color: "var(--text-secondary)" }}>
                      JPG / JPEG
                    </span>
                    <span style={{ fontSize: "0.72rem", padding: "0.2rem 0.5rem", backgroundColor: "var(--surface-secondary)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", color: "var(--text-secondary)" }}>
                      PNG
                    </span>
                    <span style={{ fontSize: "0.72rem", padding: "0.2rem 0.5rem", backgroundColor: "var(--surface-secondary)", border: "1px solid var(--border)", borderRadius: "var(--radius-sm)", color: "var(--text-secondary)" }}>
                      Max 10MB
                    </span>
                  </div>

                  <div className="dropzone-privacy-note">
                    🔒 Images are processed for this analysis and are not permanently stored.
                  </div>

                  <input
                    type="file"
                    ref={fileInputRef}
                    style={{ display: "none" }}
                    accept=".jpg,.jpeg,.png,image/jpeg,image/png"
                    onChange={(e) => {
                      if (e.target.files && e.target.files[0]) {
                        handleFileProcess(e.target.files[0]);
                      }
                    }}
                  />

                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      fileInputRef.current?.click();
                    }}
                    style={{ marginTop: "0.35rem" }}
                  >
                    Browse Files
                  </button>
                </div>
              ) : (
                /* Image Preview Box when file selected */
                <div>
                  <div className="preview-container">
                    <div
                      className="preview-image-wrapper"
                      style={{ cursor: "pointer", position: "relative" }}
                      onClick={() =>
                        setModalImage({
                          title: `Input Preview: ${selectedFile.name}`,
                          src: imagePreviewSrc,
                          desc: `${imageDimensions ? `${imageDimensions.width}×${imageDimensions.height} px` : `${(selectedFile.size / 1024).toFixed(1)} KB`}`,
                        })
                      }
                      title="Click to enlarge preview"
                    >
                      {imagePreviewSrc ? (
                        <img src={imagePreviewSrc} alt="Uploaded MRI preview" />
                      ) : (
                        <span style={{ color: "#64748b" }}>Loading preview...</span>
                      )}
                      <div className="gradcam-zoom-hint">🔍 Enlarge</div>
                    </div>

                    <div className="preview-meta">
                      <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: "160px" }}>
                        {selectedFile.name}
                      </span>
                      <span>
                        {imageDimensions ? `${imageDimensions.width}×${imageDimensions.height} px` : `${(selectedFile.size / 1024).toFixed(1)} KB`}
                      </span>
                    </div>
                  </div>

                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "0.75rem" }}>
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={handleReset}
                      disabled={status === "analyzing"}
                    >
                      Change Image
                    </button>

                    <button
                      className="btn btn-danger-outline btn-sm"
                      onClick={handleReset}
                      disabled={status === "analyzing"}
                    >
                      Remove
                    </button>
                  </div>
                </div>
              )}

              {/* Demonstration Test Samples Tray */}
              <div style={{ marginTop: "1.25rem", paddingTop: "1rem", borderTop: "1px solid var(--border)" }}>
                <div style={{ fontSize: "0.78rem", fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", marginBottom: "0.4rem" }}>
                  Demonstration Brain MRI Scans
                </div>
                <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.5rem" }}>
                  Load pre-verified academic sample scans to test multi-class classification:
                </p>
                <div className="sample-tray">
                  {SAMPLE_IMAGES.map((sample) => (
                    <button
                      key={sample.id}
                      className="sample-pill-btn"
                      onClick={() => handleSelectSample(sample.path)}
                      disabled={status === "analyzing"}
                    >
                      <span>🔬</span> {sample.expectedClass}
                    </button>
                  ))}
                </div>
              </div>

              {/* Guardrail Safety Test Tray */}
              <div style={{ marginTop: "1rem", paddingTop: "1rem", borderTop: "1px solid var(--border)" }}>
                <div style={{ fontSize: "0.78rem", fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.05em", color: "#c05621", marginBottom: "0.4rem", display: "flex", alignItems: "center", gap: "0.4rem" }}>
                  <span>🛡️</span> Guardrail Safety Test Samples
                </div>
                <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.5rem" }}>
                  Test server-side rejection on non-MRI or corrupted files (classification is blocked):
                </p>
                <div className="sample-tray">
                  {GUARDRAIL_TEST_SAMPLES.map((sample) => (
                    <button
                      key={sample.id}
                      className="sample-pill-btn"
                      style={{ borderColor: "#fbd38d", color: "#9c4221", backgroundColor: "#fffaf0" }}
                      onClick={() => handleSelectSample(sample.path)}
                      disabled={status === "analyzing"}
                    >
                      <span>⚠️</span> {sample.title}
                    </button>
                  ))}
                </div>
              </div>

              {/* Primary Action Button */}
              <div style={{ marginTop: "1.25rem" }}>
                <button
                  className="btn btn-primary btn-lg"
                  style={{ width: "100%" }}
                  disabled={!selectedFile || status === "analyzing"}
                  onClick={handleAnalyze}
                >
                  {status === "analyzing" ? (
                    <>
                      <span className="spinner-medical" style={{ width: "18px", height: "18px", borderWidth: "2px", borderTopColor: "white", display: "inline-block" }} />
                      Analyzing MRI Scan...
                    </>
                  ) : (
                    <>Analyze MRI &rarr;</>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Right Column: Guidance, Loading & Guardrail States */}
          <div>
            {/* INITIAL EMPTY STATE */}
            {status === "initial" && (
              <div className="empty-state-box">
                <div className="empty-state-icon">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <circle cx="12" cy="12" r="10" />
                    <path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20" />
                    <path d="M2 12h20" />
                  </svg>
                </div>
                <div className="empty-state-title">No Scan Analyzed Yet</div>
                <div className="empty-state-desc">
                  Upload a brain MRI scan or choose one of the demonstration samples on the left to begin the
                  multi-stage validation and classification workflow.
                </div>
                <div style={{ display: "flex", gap: "0.5rem", fontSize: "0.78rem", color: "var(--text-subtle)", marginTop: "0.5rem" }}>
                  <span>1. Upload</span> &rarr;
                  <span>2. Guardrail Verify</span> &rarr;
                  <span>3. Classify</span> &rarr;
                  <span>4. Grad-CAM</span>
                </div>
              </div>
            )}

            {/* READY STATE (Image Selected, Awaiting Analysis) */}
            {status === "ready" && (
              <div className="empty-state-box" style={{ borderColor: "var(--accent)" }}>
                <div className="empty-state-icon" style={{ color: "var(--accent)" }}>
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                </div>
                <div className="empty-state-title">Scan Ready for Analysis</div>
                <div className="empty-state-desc">
                  Image loaded and decodable. Click <strong>"Analyze MRI"</strong> to initiate the Brain MRI input guardrail,
                  EfficientNetB0 classification, and Grad-CAM explainability pipeline.
                </div>
                <button className="btn btn-primary btn-sm" onClick={handleAnalyze}>
                  Run Analysis Now &rarr;
                </button>
              </div>
            )}

            {/* ANALYZING STATE */}
            {status === "analyzing" && (
              <div className="analyzing-state-box">
                <div className="spinner-medical" />
                <h3 style={{ fontSize: "1.2rem", fontWeight: "800", color: "var(--text-primary)" }}>
                  Executing Inference Pipeline
                </h3>
                <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem", fontSize: "0.85rem", color: "var(--text-muted)" }}>
                  <div>✓ In-memory image buffer loaded (224×224)</div>
                  <div>⏳ Evaluating Brain MRI Input Guardrail...</div>
                  <div>⏳ Computing EfficientNetB0 class probabilities...</div>
                  <div>⏳ Synthesizing Grad-CAM feature heatmaps...</div>
                </div>
              </div>
            )}

            {/* GUARDRAIL REJECTED STATE: Non-MRI Detected */}
            {status === "rejected" && guardrailRejection && (
              <div
                className="card"
                style={{
                  backgroundColor: "#fff5f5",
                  borderColor: "#feb2b2",
                  borderWidth: "2px",
                }}
              >
                <div style={{ display: "flex", gap: "1rem", alignItems: "flex-start" }}>
                  <span style={{ fontSize: "2rem", lineHeight: 1 }}>🛑</span>
                  <div>
                    <div
                      style={{
                        display: "inline-block",
                        fontSize: "0.72rem",
                        fontWeight: "800",
                        textTransform: "uppercase",
                        letterSpacing: "0.06em",
                        backgroundColor: "#fed7d7",
                        color: "#9b2c2c",
                        padding: "0.2rem 0.6rem",
                        borderRadius: "var(--radius-full)",
                        marginBottom: "0.5rem",
                      }}
                    >
                      Guardrail Intercept
                    </div>
                    <h3 style={{ fontSize: "1.25rem", fontWeight: "800", color: "#742a2a", marginBottom: "0.5rem" }}>
                      Image Not Accepted: Unsupported Image
                    </h3>
                    <p style={{ fontSize: "0.9rem", color: "#742a2a", lineHeight: "1.5", marginBottom: "0.75rem" }}>
                      <strong>This application only analyzes brain MRI images.</strong> Please upload a valid brain MRI image.
                    </p>
                    <p style={{ fontSize: "0.825rem", color: "#9b2c2c", backgroundColor: "#ffffff", padding: "0.6rem 0.85rem", borderRadius: "var(--radius-sm)", border: "1px solid #fecaca" }}>
                      <em>Guardrail Detail:</em> {guardrailRejection.message}
                    </p>
                    <p style={{ fontSize: "0.78rem", color: "#742a2a", marginTop: "0.75rem" }}>
                      <strong>Safety Rule Enforced:</strong> The tumor classification model was NOT executed and no Grad-CAM was generated.
                    </p>
                    <button
                      className="btn btn-secondary btn-sm"
                      style={{ marginTop: "1rem" }}
                      onClick={handleReset}
                    >
                      Upload Another Image &rarr;
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* GUARDRAIL UNCERTAIN STATE: Borderline / Low Confidence */}
            {status === "uncertain" && guardrailRejection && (
              <div
                className="card"
                style={{
                  backgroundColor: "#fffaf0",
                  borderColor: "#fbd38d",
                  borderWidth: "2px",
                }}
              >
                <div style={{ display: "flex", gap: "1rem", alignItems: "flex-start" }}>
                  <span style={{ fontSize: "2rem", lineHeight: 1 }}>⚠️</span>
                  <div>
                    <div
                      style={{
                        display: "inline-block",
                        fontSize: "0.72rem",
                        fontWeight: "800",
                        textTransform: "uppercase",
                        letterSpacing: "0.06em",
                        backgroundColor: "#feebc8",
                        color: "#c05621",
                        padding: "0.2rem 0.6rem",
                        borderRadius: "var(--radius-full)",
                        marginBottom: "0.5rem",
                      }}
                    >
                      Verification Inconclusive
                    </div>
                    <h3 style={{ fontSize: "1.25rem", fontWeight: "800", color: "#7b341e", marginBottom: "0.5rem" }}>
                      Unable to Verify Image Type
                    </h3>
                    <p style={{ fontSize: "0.9rem", color: "#7b341e", lineHeight: "1.5", marginBottom: "0.75rem" }}>
                      For safety, classification was not performed. The image could not be confidently verified as a brain MRI.
                      Please upload a clear brain MRI image.
                    </p>
                    <p style={{ fontSize: "0.825rem", color: "#9c4221", backgroundColor: "#ffffff", padding: "0.6rem 0.85rem", borderRadius: "var(--radius-sm)", border: "1px solid #fbd38d" }}>
                      <em>Guardrail Detail:</em> {guardrailRejection.message}
                    </p>
                    <button
                      className="btn btn-secondary btn-sm"
                      style={{ marginTop: "1rem" }}
                      onClick={handleReset}
                    >
                      Try A Different Brain MRI &rarr;
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* ERROR STATE: Network / Backend Failure */}
            {status === "error" && errorMessage && (
              <div
                className="card"
                style={{
                  backgroundColor: "#fef2f2",
                  borderColor: "#fecaca",
                  borderWidth: "1.5px",
                }}
              >
                <h3 style={{ fontSize: "1.1rem", fontWeight: "700", color: "var(--error)", marginBottom: "0.5rem" }}>
                  Analysis Request Notice
                </h3>
                <p style={{ fontSize: "0.875rem", color: "#7f1d1d", lineHeight: "1.5" }}>
                  {errorMessage}
                </p>
                <button
                  className="btn btn-secondary btn-sm"
                  style={{ marginTop: "1rem" }}
                  onClick={() => setStatus("initial")}
                >
                  Dismiss & Try Again
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Input Scan Zoom Modal */}
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
