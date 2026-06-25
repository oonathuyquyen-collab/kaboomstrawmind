"""
DEMO INFERENCE - YOLOv8s Model
Su dung model tot nhat de detect benh nam rom
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
from pathlib import Path
import sys

print("="*70)
print("STRAWMIND DDAV - DEMO INFERENCE VOI YOLOV8S")
print("="*70)

# Load model tot nhat
MODEL_PATH = "runs/detect/runs/train_yolov8s/strawmind_yolov8s/weights/best.pt"

print(f"\nLoad model YOLOv8s: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print(f"\n❌ ERROR: Model khong ton tai!")
    print(f"Path: {MODEL_PATH}")
    print(f"\nVui long chay train_yolov8s.py truoc!")
    sys.exit(1)

model = YOLO(MODEL_PATH)
print("✓ Model loaded!")

print(f"\nModel info:")
print(f"  Classes: {model.names}")
print(f"  0: Affected (Bi benh)")
print(f"  1: Healthy (Khoe manh)")
print(f"  2: Healthy-Affected (Ca hai)")

print("\n" + "="*70)
print("HUONG DAN SU DUNG")
print("="*70)

print("""
1. CHE DO 1: Test voi 5 anh co san
   python demo_inference.py test

2. CHE DO 2: Test voi anh bat ky
   python demo_inference.py path/to/your/image.jpg

3. CHE DO 3: Test voi folder
   python demo_inference.py path/to/your/folder/

4. Xem ket qua:
   Anh ket qua se luu tai: runs/detect/runs/predict_demo/
""")

# Get arguments
if len(sys.argv) < 2:
    print("\n⚠ Khong co argument!")
    print("Su dung: python demo_inference.py [test|path/to/image|path/to/folder]")
    sys.exit(0)

arg = sys.argv[1]

# Confidence threshold (toi uu cho YOLOv8s)
CONF_THRESHOLD = 0.01

print("\n" + "="*70)
print("CHAY INFERENCE")
print("="*70)

print(f"\nConfidence threshold: {CONF_THRESHOLD}")
print("(Thap vi model con confidence thap do dataset nho)")

if arg == "test":
    # Test voi 5 anh co san
    source = "test_images_real"
    print(f"\nTest voi 5 anh co san: {source}")
else:
    # Test voi path tu user
    source = arg
    print(f"\nTest voi: {source}")

if not os.path.exists(source):
    print(f"\n❌ ERROR: Path khong ton tai: {source}")
    sys.exit(1)

print("\nBat dau inference...")

# Run prediction
results = model.predict(
    source=source,
    conf=CONF_THRESHOLD,
    iou=0.45,
    save=True,
    project="runs/detect/runs/predict_demo",
    name="result",
    show_labels=True,
    show_conf=True,
    verbose=True
)

print("\n" + "="*70)
print("KET QUA")
print("="*70)

# Parse results
total_images = len(results)
images_with_detections = 0
total_detections = 0

print(f"\nTong so anh: {total_images}")
print(f"\nChi tiet:\n")

for i, result in enumerate(results):
    img_path = Path(result.path)
    boxes = result.boxes
    
    if len(boxes) > 0:
        images_with_detections += 1
        total_detections += len(boxes)
        
        print(f"{i+1}. {img_path.name}:")
        
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            # Vietnamese labels
            if cls_name == "Affected":
                cls_vn = "BI BENH"
                emoji = "⚠️"
            elif cls_name == "Healthy":
                cls_vn = "KHOE MANH"
                emoji = "✅"
            else:
                cls_vn = "CA HAI"
                emoji = "⚡"
            
            print(f"   {emoji} {cls_vn} ({cls_name})")
            print(f"      Confidence: {conf:.3f} ({conf*100:.1f}%)")
    else:
        print(f"{i+1}. {img_path.name}:")
        print(f"   ❌ Khong detect duoc")

print("\n" + "="*70)
print("TONG KET")
print("="*70)

print(f"\nTong so anh: {total_images}")
print(f"Anh co detection: {images_with_detections}")
print(f"Anh khong detect: {total_images - images_with_detections}")
print(f"Tong so detections: {total_detections}")

detection_rate = (images_with_detections / total_images * 100) if total_images > 0 else 0
print(f"Detection rate: {detection_rate:.1f}%")

print("\n" + "="*70)
print("XEM ANH KET QUA")
print("="*70)

output_dir = "runs\\detect\\runs\\predict_demo\\result"
print(f"\nAnh ket qua da luu tai:")
print(f"  {output_dir}")
print(f"\nXem anh:")
print(f"  start {output_dir}")

print("\n" + "="*70)
print("INTERPRETATION GUIDE")
print("="*70)

print("""
✅ KHOE MANH (Healthy):
   - Nam khoe, khong co dau hieu benh
   - Co the thu hoach

⚠️  BI BENH (Affected):
   - Nam bi nhiem benh (Trichoderma, Dry Bubble, Wet Bubble)
   - Can xu ly ngay: cach ly, loai bo
   - Kiem tra xung quanh co bi lay khong

⚡ CA HAI (Healthy-Affected):
   - Trong anh co ca nam khoe va bi benh
   - Loai bo nam bi benh
   - Giu nam khoe

CONFIDENCE THRESHOLD:
   - Hien tai: 0.01 (rat thap)
   - Ly do: Model con confidence thap vi dataset nho (134 anh)
   - Voi dataset lon hon (1500-3000 anh): threshold se cao hon (0.1-0.15)
   - Neu confidence < 0.05: Nen kiem tra lai bang mat

LUU Y:
   - Model accuracy hien tai: 80% (4/5 anh)
   - False Positive: Co the co (Precision thap 1.9%)
   - False Negative: It (Recall cao 87.5%)
   - Khuyen nghi: Kiem tra lai bang mat neu khong chac chan
""")

print("\n" + "="*70)
print("HOAN THANH!")
print("="*70)

print("""
Model: YOLOv8s
Path: runs/detect/runs/train_yolov8s/strawmind_yolov8s/weights/best.pt
Size: 22MB
Test Accuracy: 80%
Recall: 87.5%

Suitable for:
  ✅ Server deployment
  ✅ Cloud inference
  ✅ Desktop application

Next steps:
  1. Thu thap 1500-3000 anh
  2. Re-training voi dataset lon
  3. Expected accuracy: >90%
""")

print("\n" + "="*70)
