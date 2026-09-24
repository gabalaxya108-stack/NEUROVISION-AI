"""
Creates a controlled test suite of 9 representative input cases for the Brain MRI Guardrail:
1. Valid brain MRI
2. Non-MRI photograph
3. Document / text page
4. Screenshot
5. Non-brain X-ray radiograph
6. Non-brain CT / abdominal scan
7. Random pattern
8. Very low-quality thumbnail (<64px)
9. Corrupted byte stream
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

OUTPUT_DIR = Path("tests/guardrail_samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Valid Brain MRI (Curated sample)
source_mri = Path("samples/sample_glioma.jpg")
if source_mri.exists():
    with Image.open(source_mri) as img:
        img.save(OUTPUT_DIR / "01_valid_brain_mri.jpg")

# 2. Non-MRI Photograph (Colorful landscape / selfie simulation)
photo = Image.new("RGB", (300, 300))
draw = ImageDraw.Draw(photo)
# Blue sky, green grass, golden sun
draw.rectangle([0, 0, 300, 180], fill=(70, 130, 240))  # Vivid blue sky
draw.rectangle([0, 180, 300, 300], fill=(34, 139, 34))  # Rich green grass
draw.ellipse([200, 30, 270, 100], fill=(255, 204, 0))   # Bright yellow sun
photo.save(OUTPUT_DIR / "02_photograph.jpg")

# 3. Document / Text Page (White background, black text lines)
doc = Image.new("RGB", (300, 380), color=(255, 255, 255))
draw_doc = ImageDraw.Draw(doc)
# Draw text lines
for y in range(40, 340, 18):
    draw_doc.rectangle([30, y, 270, y + 4], fill=(30, 30, 30))
doc.save(OUTPUT_DIR / "03_document.png")

# 4. Computer Screenshot / UI (Light gray background, buttons, rectilinear windows)
screenshot = Image.new("RGB", (400, 300), color=(240, 242, 245))
draw_ss = ImageDraw.Draw(screenshot)
# Top window bar
draw_ss.rectangle([10, 10, 390, 40], fill=(45, 55, 72))
# UI buttons & rectangles
draw_ss.rectangle([20, 55, 180, 140], fill=(255, 255, 255), outline=(200, 200, 200), width=2)
draw_ss.rectangle([200, 55, 380, 140], fill=(255, 255, 255), outline=(200, 200, 200), width=2)
draw_ss.rectangle([20, 160, 380, 280], fill=(255, 255, 255), outline=(200, 200, 200), width=2)
screenshot.save(OUTPUT_DIR / "04_screenshot.png")

# 5. Non-Brain X-Ray (Chest radiograph simulation - bright ribs and clavicles extending to borders)
xray_arr = np.zeros((300, 300), dtype=np.uint8)
# Light border corners typical of chest radiograph plates
xray_arr[:40, :] = 160
xray_arr[-40:, :] = 160
# Vertical spinal column & lateral rib cage
for i in range(50, 250, 25):
    xray_arr[i : i + 8, 30:270] = 210
xray_img = Image.fromarray(xray_arr).convert("RGB")
xray_img.save(OUTPUT_DIR / "05_xray.jpg")

# 6. Non-Brain CT (Abdominal/pelvic slice simulation filling whole canvas without dark air perimeter)
ct_arr = np.full((300, 300), 120, dtype=np.uint8)
# Full-bleed subcutaneous tissue
ct_arr[30:270, 30:270] = 80
ct_arr[80:220, 80:220] = 160
ct_img = Image.fromarray(ct_arr).convert("RGB")
ct_img.save(OUTPUT_DIR / "06_ct_abdomen.jpg")

# 7. Random Colorful Pattern
np.random.seed(42)
rand_arr = np.random.randint(0, 256, (250, 250, 3), dtype=np.uint8)
rand_img = Image.fromarray(rand_arr)
rand_img.save(OUTPUT_DIR / "07_random_pattern.jpg")

# 8. Very Low-Quality Thumbnail (<64px)
tiny_img = Image.new("RGB", (32, 32), color=(50, 50, 50))
tiny_img.save(OUTPUT_DIR / "08_low_quality_tiny.jpg")

# 9. Corrupted Byte Stream
corrupt_path = OUTPUT_DIR / "09_corrupted.jpg"
with open(corrupt_path, "wb") as f:
    f.write(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00\x60\x00\x60\x00\x00CORRUPTED_TRUNCATED_DATA_STREAM")

print("Created 9 test cases in tests/guardrail_samples:")
for p in sorted(OUTPUT_DIR.glob("*")):
    print(f" - {p.name}")
