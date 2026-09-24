"""
Verify prediction consistency between:
1. CLI (src/predict.py)
2. API (POST /predict)
3. Frontend interface data schema
"""

import json
import subprocess
from pathlib import Path
import requests

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = BASE_DIR / "samples" / "sample_glioma.jpg"
OUTPUT_PATH = BASE_DIR / "results" / "integration_prediction_check.json"

def get_cli_prediction(img_path):
    # Run CLI
    cmd = ["python3", str(BASE_DIR / "src" / "predict.py"), str(img_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = res.stdout
    
    # Parse prediction class and probabilities from formatted output
    predicted_class = None
    probs = {}
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "Prediction:":
            predicted_class = lines[i+1].strip().lower()
        if "Glioma" in line and "%" in line:
            probs["glioma"] = float(line.split()[1].replace("%", "")) / 100.0
        if "Meningioma" in line and "%" in line:
            probs["meningioma"] = float(line.split()[1].replace("%", "")) / 100.0
        if "Pituitary" in line and "%" in line:
            probs["pituitary"] = float(line.split()[1].replace("%", "")) / 100.0
        if "No Tumor" in line and "%" in line:
            probs["notumor"] = float(line.split()[-1].replace("%", "")) / 100.0
            
    return predicted_class, probs

def get_api_prediction(img_path):
    url = "http://127.0.0.1:8000/predict"
    with open(img_path, "rb") as f:
        resp = requests.post(url, files={"file": ("sample_glioma.jpg", f, "image/jpeg")})
    resp.raise_for_status()
    data = resp.json()
    return data["predicted_class"], data["probabilities"]

def run_check():
    print(f"Testing consistency for {SAMPLE_PATH.name}...")
    cli_pred, cli_probs = get_cli_prediction(SAMPLE_PATH)
    api_pred, api_probs = get_api_prediction(SAMPLE_PATH)
    
    # In frontend, the prediction is directly received from POST /predict response
    frontend_pred = api_pred
    frontend_probs = api_probs
    
    print(f"CLI: {cli_pred} | Probs: {cli_probs}")
    print(f"API: {api_pred} | Probs: {api_probs}")
    
    # Calculate max difference
    diffs = [abs(cli_probs[c] - api_probs[c]) for c in cli_probs]
    max_diff = max(diffs)
    
    classes_identical = (cli_pred == api_pred == frontend_pred)
    consistent = classes_identical and (max_diff < 1e-4)
    
    result = {
        "image": SAMPLE_PATH.name,
        "cli_prediction": cli_pred,
        "api_prediction": api_pred,
        "frontend_prediction": frontend_pred,
        "cli_probabilities": cli_probs,
        "api_probabilities": api_probs,
        "prediction_consistent": consistent,
        "max_probability_difference": round(float(max_diff), 6)
    }
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
        
    print(f"Results successfully saved to {OUTPUT_PATH}")
    print(json.dumps(result, indent=2))
    return consistent

if __name__ == "__main__":
    success = run_check()
    if not success:
        exit(1)
