"""
Test invalid inputs against the API:
- text file
- corrupted image
- unsupported file format
- empty upload
Verify user-friendly error, no stack trace, no server crash.
"""

import requests
from pathlib import Path

URL = "http://127.0.0.1:8000/predict"

def test_invalid():
    test_cases = [
        ("Text file (.txt)", ("notes.txt", b"This is just a plain text document.", "text/plain"), [400, 415, 422]),
        ("Corrupted image", ("corrupt.jpg", b"NOT_A_REAL_JPEG_IMAGE_HEADER_XYZ", "image/jpeg"), [400, 422]),
        ("Unsupported format (.pdf)", ("document.pdf", b"%PDF-1.4 dummy pdf bytes", "application/pdf"), [400, 415, 422]),
        ("Empty file (0 bytes)", ("empty.jpg", b"", "image/jpeg"), [400, 422]),
    ]
    
    print("Testing Invalid Inputs...")
    results = []
    for name, (filename, content, mime), expected_statuses in test_cases:
        resp = requests.post(URL, files={"file": (filename, content, mime)})
        print(f"[{name}] HTTP Status: {resp.status_code}")
        
        # Check that response is valid JSON and does not contain tracebacks or internal paths
        try:
            body = resp.json()
            print(f"  Response JSON: {body}")
            body_str = str(body).lower()
            has_traceback = "traceback" in body_str or "file \"" in body_str or "line " in body_str
        except Exception:
            body = resp.text
            print(f"  Response text: {body}")
            has_traceback = "traceback" in body.lower()
            
        status_ok = resp.status_code in expected_statuses
        safe_response = not has_traceback
        passed = status_ok and safe_response
        
        print(f"  Status Match: {status_ok} | Safe (No Stack Trace): {safe_response}")
        results.append({
            "name": name,
            "status_code": resp.status_code,
            "passed": passed
        })
        
    all_passed = all(r["passed"] for r in results)
    print(f"\nAll invalid input tests passed: {all_passed}")
    return all_passed

if __name__ == "__main__":
    success = test_invalid()
    if not success:
        exit(1)
