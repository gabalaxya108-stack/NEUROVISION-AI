"""
Benchmark performance timings for:
- API health response
- Prediction response time
- Grad-CAM response time
- Frontend initial bundle response
"""

import time
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
HEALTH_URL = "http://127.0.0.1:8000/health"
PREDICT_URL = "http://127.0.0.1:8000/predict"
EXPLAIN_URL = "http://127.0.0.1:8000/explain"
FRONTEND_URL = "http://127.0.0.1:5173"
SAMPLE_PATH = BASE_DIR / "samples" / "sample_glioma.jpg"

def benchmark():
    with open(SAMPLE_PATH, "rb") as f:
        img_bytes = f.read()

    # 1. Frontend bundle response
    fe_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        resp = requests.get(FRONTEND_URL)
        fe_times.append((time.perf_counter() - t0) * 1000)
    
    # 2. Health check response
    health_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        resp = requests.get(HEALTH_URL)
        health_times.append((time.perf_counter() - t0) * 1000)

    # 3. Prediction response time
    pred_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        resp = requests.post(PREDICT_URL, files={"file": ("sample.jpg", img_bytes, "image/jpeg")})
        pred_times.append((time.perf_counter() - t0) * 1000)

    # 4. Grad-CAM response time
    explain_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        resp = requests.post(EXPLAIN_URL, files={"file": ("sample.jpg", img_bytes, "image/jpeg")})
        explain_times.append((time.perf_counter() - t0) * 1000)

    def stats(times):
        return {
            "mean_ms": round(sum(times) / len(times), 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2)
        }

    results = {
        "frontend_initial_load": stats(fe_times),
        "api_health_response": stats(health_times),
        "prediction_response_time": stats(pred_times),
        "gradcam_response_time": stats(explain_times)
    }

    print("Performance Measurement Results:")
    for k, v in results.items():
        print(f"  {k}: Mean = {v['mean_ms']} ms (Min: {v['min_ms']} ms, Max: {v['max_ms']} ms)")

    return results

if __name__ == "__main__":
    benchmark()
