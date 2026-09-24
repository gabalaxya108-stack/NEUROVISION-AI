/**
 * Brain MRI Tumor Classification - Constants & Educational Metadata
 */

export const API_BASE_URL =
  import.meta.env.VITE_API_URL !== undefined
    ? import.meta.env.VITE_API_URL
    : (import.meta.env.DEV ? "http://127.0.0.1:8000" : "");

export const CLASSES = ["glioma", "meningioma", "pituitary", "notumor"];

export const CLASS_METADATA = {
  glioma: {
    id: 0,
    name: "Glioma",
    description: "Tumor originating from glial cells within brain parenchyma (intra-axial).",
    badgeColor: "#2563eb",
    bgColor: "#eff6ff",
    borderColor: "#bfdbfe",
  },
  meningioma: {
    id: 1,
    name: "Meningioma",
    description: "Slow-growing tumor arising from meningeal coverings of brain and spinal cord (extra-axial).",
    badgeColor: "#7c3aed",
    bgColor: "#f5f3ff",
    borderColor: "#ddd6fe",
  },
  pituitary: {
    id: 2,
    name: "Pituitary",
    description: "Benign adenoma originating in the sella turcica / pituitary gland region.",
    badgeColor: "#0d9488",
    bgColor: "#f0fdfa",
    borderColor: "#99f6e4",
  },
  notumor: {
    id: 3,
    name: "No Tumor",
    description: "Normal, healthy brain tissue appearance with intact neuroanatomy.",
    badgeColor: "#16a34a",
    bgColor: "#f0fdf4",
    borderColor: "#bbf7d0",
  },
};

export const PROJECT_STATS = [
  {
    value: "7,200",
    label: "MRI Images",
    detail: "5,600 training & 1,600 held-out test scans",
  },
  {
    value: "4",
    label: "Classes",
    detail: "Glioma, Meningioma, Pituitary, No Tumor",
  },
  {
    value: "92.81%",
    label: "Test-set classification accuracy",
    detail: "Empirically measured on untouched test set (N=1,600)",
  },
  {
    value: "EfficientNetB0",
    label: "Architecture",
    detail: "Transfer learning with Grad-CAM explainability",
  },
];

export const PER_CLASS_PERFORMANCE = [
  {
    className: "Glioma",
    precision: "97.20%",
    recall: "78.00%",
    f1: "86.55%",
    total: 400,
    correct: 312,
    notes: "High precision (rarely false-alarms), lower recall due to border mimicry with meningiomas.",
  },
  {
    className: "Meningioma",
    precision: "86.64%",
    recall: "94.00%",
    f1: "90.17%",
    total: 400,
    correct: 376,
    notes: "High recall (catches most meningiomas), receives 56 misclassified glioma cases.",
  },
  {
    className: "Pituitary",
    precision: "95.69%",
    recall: "100.00%",
    f1: "97.80%",
    total: 400,
    correct: 400,
    notes: "Flawless recall (0 false negatives); distinctive anatomical location in sella turcica.",
  },
  {
    className: "No Tumor",
    precision: "92.97%",
    recall: "99.25%",
    f1: "96.01%",
    total: 400,
    correct: 397,
    notes: "99.25% recall (only 3 false positives out of 400 normal scans).",
  },
];

export const SAMPLE_IMAGES = [
  {
    id: "demo",
    title: "Sample 1: Healthy Scan",
    expectedClass: "No Tumor",
    path: "/samples/demo_external_mri.jpg",
  },
  {
    id: "glioma",
    title: "Sample 2: Glioma Scan",
    expectedClass: "Glioma",
    path: "/samples/sample_glioma.jpg",
  },
  {
    id: "meningioma",
    title: "Sample 3: Meningioma Scan",
    expectedClass: "Meningioma",
    path: "/samples/sample_meningioma.jpg",
  },
  {
    id: "pituitary",
    title: "Sample 4: Pituitary Scan",
    expectedClass: "Pituitary",
    path: "/samples/sample_pituitary.jpg",
  },
  {
    id: "notumor",
    title: "Sample 5: Normal Control",
    expectedClass: "No Tumor",
    path: "/samples/sample_notumor.jpg",
  },
];

export const GUARDRAIL_TEST_SAMPLES = [
  {
    id: "guard_photo",
    title: "Non-MRI Photo",
    expectedResult: "REJECTED (non_brain_mri)",
    path: "/samples/test_photograph.jpg",
  },
  {
    id: "guard_doc",
    title: "Document / Text",
    expectedResult: "REJECTED (non_brain_mri)",
    path: "/samples/test_document.png",
  },
  {
    id: "guard_ss",
    title: "UI Screenshot",
    expectedResult: "REJECTED (non_brain_mri)",
    path: "/samples/test_screenshot.png",
  },
];

export const REPRESENTATIVE_ERRORS = [
  {
    filename: "Te-gl_109.jpg",
    actual: "Glioma",
    predicted: "Meningioma",
    confidence: "99.93%",
    imagePath: "/results/error_analysis/top_misclassified.png",
    gradcamPath: "/results/gradcam/errors/error_gradcam_Te-gl_109.png",
    analysis: "Dense peripheral enhancement mimicking dural attachment typical of extra-axial meningiomas.",
  },
  {
    filename: "Te-gl_160.jpg",
    actual: "Glioma",
    predicted: "Meningioma",
    confidence: "99.87%",
    imagePath: "/results/error_analysis/high_confidence_errors.png",
    gradcamPath: "/results/gradcam/errors/error_gradcam_Te-gl_160.png",
    analysis: "Circumscribed contrast rim along cranial vault triggering meningioma activation patterns.",
  },
  {
    filename: "Te-gl_144.jpg",
    actual: "Glioma",
    predicted: "No Tumor",
    confidence: "98.92%",
    imagePath: "/results/error_analysis/borderline_predictions.png",
    gradcamPath: "/results/gradcam/errors/error_gradcam_Te-gl_144.png",
    analysis: "Faint low-grade infiltrative signal without sharp mass effect, misread as normal parenchymal symmetry.",
  },
  {
    filename: "Te-aug-me_24.jpg",
    actual: "Meningioma",
    predicted: "Pituitary",
    confidence: "95.42%",
    imagePath: "/results/error_analysis/top_misclassified.png",
    gradcamPath: "/results/gradcam/errors/error_gradcam_Te-aug-me_24.png",
    analysis: "Parasellar mass location near sphenoid wing confounding regional anatomical cues.",
  },
];

export const DISCLAIMER_TEXT =
  "This application is intended for educational and research purposes only. It is not a medical diagnostic system and should not be used for clinical decision-making.";
