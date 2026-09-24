"""
Test multiple image formats (Grayscale, RGB, PNG, JPG, different dimensions)
on the Brain MRI classification inference API.
"""

import io
import requests
from PIL import Image
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8000/predict"

def test_formats():
    # Use existing genuine MRI sample as base to preserve valid brain MRI structure
    base_sample_path = BASE_DIR / "samples" / "sample_glioma.jpg"
    base_img = Image.open(base_sample_path)
    
    test_cases = [
        ("RGB JPEG", base_img.convert("RGB"), "image/jpeg", "test_rgb.jpg"),
        ("Grayscale JPEG (L mode)", base_img.convert("L"), "image/jpeg", "test_gray.jpg"),
        ("RGB PNG", base_img.convert("RGB"), "image/png", "test_rgb.png"),
        ("Grayscale PNG (L mode)", base_img.convert("L"), "image/png", "test_gray.png"),
        ("Different Dimensions (384x512)", base_img.resize((384, 512)), "image/jpeg", "test_384x512.jpg"),
        ("Square High-Res (600x600)", base_img.resize((600, 600)), "image/png", "test_600x600.png"),
    ]
    
    results = []
    print("Testing Multiple Image Formats & Dimensions...")
    
    for name, img, mime, filename in test_cases:
        buf = io.BytesIO()
        fmt = "PNG" if "png" in filename.lower() else "JPEG"
        img.save(buf, format=fmt)
        buf.seek(0)
        
        resp = requests.post(URL, files={"file": (filename, buf, mime)})
        print(f"[{name}] Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Accepted: {data.get('accepted')} | Pred: {data.get('predicted_class')} | Conf: {data.get('confidence')}")
            results.append({
                "format_test": name,
                "status_code": resp.status_code,
                "accepted": data.get("accepted"),
                "predicted_class": data.get("predicted_class"),
                "confidence": data.get("confidence"),
                "success": data.get("accepted") is True
            })
        else:
            print(f"  Failed: {resp.text}")
            results.append({
                "format_test": name,
                "status_code": resp.status_code,
                "success": False
            })
            
    all_success = all(r["success"] for r in results)
    print(f"\nAll format tests passed: {all_success}")
    return all_success, results

if __name__ == "__main__":
    success, res = test_formats()
    if not success:
        exit(1)
