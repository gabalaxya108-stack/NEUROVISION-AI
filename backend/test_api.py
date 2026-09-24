"""
Comprehensive Backend API Test Suite for Brain MRI Classification.
Validates:
1. GET /health
2. GET /model-info
3. POST /predict with a controlled sample image
4. Comparison of POST /predict against src/predict.py (parity check)
5. POST /predict with invalid / corrupted file (error handling check)
6. POST /explain with a controlled sample image (Grad-CAM visualization check)
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Ensure torch backend for Keras
os.environ.setdefault("KERAS_BACKEND", "torch")

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from src.predict import BrainTumorPredictor
from src.config import BEST_MODEL_PATH


def run_all_tests():
    print("=" * 60)
    print("STARTING BACKEND API TEST SUITE")
    print("=" * 60)

    # Initialize TestClient (triggers lifespan context to load model)
    print("\n[STEP 1] Initializing TestClient and triggering model loading...")
    with TestClient(app) as client:
        # 1. Test /health
        print("\n[TEST 1] Testing GET /health...")
        res_health = client.get("/health")
        print(f"Status Code: {res_health.status_code}")
        print(f"Response: {res_health.json()}")
        assert res_health.status_code == 200, f"Expected 200, got {res_health.status_code}"
        health_data = res_health.json()
        assert health_data["status"] == "ok"
        assert health_data["model_loaded"] is True
        print("✓ GET /health PASSED")

        # 2. Test /model-info
        print("\n[TEST 2] Testing GET /model-info...")
        res_info = client.get("/model-info")
        print(f"Status Code: {res_info.status_code}")
        print(f"Response: {res_info.json()}")
        assert res_info.status_code == 200, f"Expected 200, got {res_info.status_code}"
        info_data = res_info.json()
        assert info_data["model"] == "EfficientNetB0"
        assert info_data["input_shape"] == [224, 224, 3]
        assert info_data["classes"] == ["glioma", "meningioma", "pituitary", "notumor"]
        assert info_data["test_accuracy"] == 0.9281
        assert "educational/research" in info_data["purpose"]
        print("✓ GET /model-info PASSED")

        # 3. Test /predict with a controlled sample image
        sample_img_path = PROJECT_ROOT / "samples" / "demo_external_mri.jpg"
        if not sample_img_path.exists():
            # Fallback to test image if demo sample missing
            sample_img_path = PROJECT_ROOT / "archive" / "Testing" / "notumor" / "Te-no_0010.jpg"
        
        print(f"\n[TEST 3] Testing POST /predict with sample: {sample_img_path.name}...")
        with open(sample_img_path, "rb") as f:
            file_bytes = f.read()

        files = {"file": (sample_img_path.name, file_bytes, "image/jpeg")}
        res_pred = client.post("/predict", files=files)
        print(f"Status Code: {res_pred.status_code}")
        pred_data = res_pred.json()
        print(f"Response: {pred_data}")
        assert res_pred.status_code == 200, f"Expected 200, got {res_pred.status_code}: {pred_data}"
        assert "predicted_class" in pred_data
        assert "confidence" in pred_data
        assert "probabilities" in pred_data
        assert len(pred_data["probabilities"]) == 4
        print("✓ POST /predict PASSED")

        # 4. Parity Verification between src/predict.py and POST /predict
        print("\n[TEST 4] Verifying inference parity with standalone src/predict.py...")
        standalone_predictor = BrainTumorPredictor(model_path=BEST_MODEL_PATH)
        standalone_result = standalone_predictor.predict(sample_img_path)
        
        standalone_pred_class = standalone_result["predicted_class"]
        api_pred_class = pred_data["predicted_class"]
        print(f"Standalone predicted class: {standalone_pred_class}")
        print(f"API predicted class:        {api_pred_class}")
        assert standalone_pred_class == api_pred_class, (
            f"Class mismatch: Standalone={standalone_pred_class}, API={api_pred_class}"
        )

        for cls_name in ["glioma", "meningioma", "pituitary", "notumor"]:
            std_prob = standalone_result["probabilities"][cls_name]
            api_prob = pred_data["probabilities"][cls_name]
            diff = abs(std_prob - api_prob)
            print(f"  Class '{cls_name:10s}': Standalone={std_prob:.4f}, API={api_prob:.4f} (diff={diff:.6f})")
            assert diff < 1e-3, f"Score discrepancy on {cls_name}: diff={diff}"
        print("✓ Inference Parity PASSED (Scores are identical within floating point tolerance)")

        # 5. Test error handling on invalid files
        print("\n[TEST 5] Testing error handling with invalid/corrupted files...")
        
        # 5a. Non-image text file
        invalid_txt_files = {"file": ("test.txt", b"This is a plain text file, not an image.", "text/plain")}
        res_txt = client.post("/predict", files=invalid_txt_files)
        print(f"Text file upload status: {res_txt.status_code} | response: {res_txt.json()}")
        assert res_txt.status_code in (400, 415), f"Expected 400 or 415, got {res_txt.status_code}"
        assert "error" in res_txt.json()

        # 5b. Corrupted fake image file
        corrupted_files = {"file": ("corrupt.jpg", b"JFIF\x00\x00\x00not-a-valid-jpeg-image", "image/jpeg")}
        res_corrupt = client.post("/predict", files=corrupted_files)
        print(f"Corrupted image upload status: {res_corrupt.status_code} | response: {res_corrupt.json()}")
        assert res_corrupt.status_code in (400, 415), f"Expected 400, got {res_corrupt.status_code}"
        assert "error" in res_corrupt.json()

        # 5c. Empty file
        empty_files = {"file": ("empty.jpg", b"", "image/jpeg")}
        res_empty = client.post("/predict", files=empty_files)
        print(f"Empty file upload status: {res_empty.status_code} | response: {res_empty.json()}")
        assert res_empty.status_code == 400, f"Expected 400, got {res_empty.status_code}"
        assert "error" in res_empty.json()
        print("✓ Error Handling PASSED")

        # 6. Test /explain endpoint
        print(f"\n[TEST 6] Testing POST /explain with sample: {sample_img_path.name}...")
        with open(sample_img_path, "rb") as f:
            file_bytes = f.read()

        files = {"file": (sample_img_path.name, file_bytes, "image/jpeg")}
        res_explain = client.post("/explain", files=files)
        print(f"Status Code: {res_explain.status_code}")
        assert res_explain.status_code == 200, f"Expected 200, got {res_explain.status_code}: {res_explain.text}"
        explain_data = res_explain.json()
        
        assert "predicted_class" in explain_data
        assert "confidence" in explain_data
        assert "probabilities" in explain_data
        assert "gradcam_overlay_base64" in explain_data
        assert "gradcam_heatmap_base64" in explain_data
        assert "gradcam_panel_base64" in explain_data

        assert explain_data["gradcam_overlay_base64"].startswith("data:image/png;base64,")
        assert explain_data["gradcam_heatmap_base64"].startswith("data:image/png;base64,")
        assert explain_data["gradcam_panel_base64"].startswith("data:image/png;base64,")
        print(f"Predicted class: {explain_data['predicted_class']} (confidence: {explain_data['confidence']:.4f})")
        print(f"Overlay base64 size: {len(explain_data['gradcam_overlay_base64'])} chars")
        print(f"Heatmap base64 size: {len(explain_data['gradcam_heatmap_base64'])} chars")
        print(f"Panel figure base64 size: {len(explain_data['gradcam_panel_base64'])} chars")
        print("✓ POST /explain PASSED")

    print("\n" + "=" * 60)
    print("ALL 6 BACKEND API TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1)
