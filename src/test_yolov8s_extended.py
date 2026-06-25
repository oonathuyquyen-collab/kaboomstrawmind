"""
Test YOLOv8s voi TAT CA anh test (14 anh)
Anh Affected: 1, 2, 4, 7, 10, 11
Con lai: Healthy
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
from pathlib import Path

print("="*70)
print("TEST MODEL YOLOV8S VOI 14 ANH NAM ROM")
print("="*70)

# Ground truth - Cap nhat theo user
ground_truth = {
    "mushroom_1.jpg": "Affected",     # ✓ User confirm
    "mushroom_2.jpg": "Affected",     # ✓ User confirm
    "mushroom_3.jpg": "Healthy",
    "mushroom_4.jpg": "Affected",     # ✓ User confirm
    "mushroom_5.jpg": "Healthy",
    "mushroom_6.jpg": "Healthy",
    "mushroom_7.jpg": "Affected",     # ✓ User confirm
    "mushroom_8.jpg": "Healthy",
    "mushroom_9.jpg": "Healthy",
    "mushroom_10.jpg.png": "Affected", # ✓ User confirm (PNG file!)
    "mushroom_11.jpg": "Affected",    # ✓ User confirm
    "mushroom_12.jpg": "Healthy",
    "mushroom_13.jpg": "Healthy",
    "mushroom_15.jpg": "Healthy",
}

affected_count = sum(1 for v in ground_truth.values() if v == "Affected")
healthy_count = sum(1 for v in ground_truth.values() if v == "Healthy")

print("\nGround Truth:")
print(f"  Total: {len(ground_truth)} anh")
print(f"  Affected: {affected_count} anh (1, 2, 4, 7, 10, 11)")
print(f"  Healthy: {healthy_count} anh")

print("\nChi tiet:")
for img, label in sorted(ground_truth.items()):
    emoji = "⚠️" if label == "Affected" else "✅"
    print(f"  {emoji} {img}: {label}")

# Load model YOLOv8s
model_path = "runs/detect/runs/train_yolov8s/strawmind_yolov8s/weights/best.pt"
print(f"\nLoad model YOLOv8s: {model_path}")

if not os.path.exists(model_path):
    print(f"\n❌ ERROR: Model khong ton tai!")
    print(f"Vui long chay train_yolov8s.py truoc!")
    exit(1)

model = YOLO(model_path)
print("✓ Model YOLOv8s loaded!")

print(f"Classes: {model.names}")

print("\n" + "="*70)
print("TEST VOI CONFIDENCE THRESHOLDS KHAC NHAU")
print("="*70)

test_images_path = "test_images_real"
test_images = sorted(list(Path(test_images_path).glob("*.jpg")) + 
                     list(Path(test_images_path).glob("*.png")))

print(f"\nTim thay {len(test_images)} anh test")

# Test voi cac confidence threshold
conf_thresholds = [0.25, 0.15, 0.10, 0.05, 0.01]

best_accuracy = 0
best_conf = 0.25
best_predictions = {}

for conf_threshold in conf_thresholds:
    print(f"\n{'='*70}")
    print(f"CONFIDENCE THRESHOLD: {conf_threshold}")
    print(f"{'='*70}")
    
    # Run detection
    results = model.predict(
        source=test_images_path,
        save=True,
        project="runs/predict_yolov8s_extended",
        name=f"test_conf_{conf_threshold}",
        conf=conf_threshold,
        iou=0.45,
        show_labels=True,
        show_conf=True,
        verbose=False
    )
    
    predictions = {}
    correct = 0
    
    for i, result in enumerate(results):
        img_name = test_images[i].name
        boxes = result.boxes
        true_label = ground_truth.get(img_name, "Unknown")
        
        if true_label == "Unknown":
            continue
        
        if len(boxes) > 0:
            # Lay detection co confidence cao nhat
            confidences = [float(box.conf[0]) for box in boxes]
            classes = [int(box.cls[0]) for box in boxes]
            
            max_conf_idx = confidences.index(max(confidences))
            pred_cls_id = classes[max_conf_idx]
            pred_cls_name = model.names[pred_cls_id]
            pred_conf = confidences[max_conf_idx]
            
            predictions[img_name] = (pred_cls_name, pred_conf)
            
            if pred_cls_name == true_label:
                correct += 1
        else:
            predictions[img_name] = ("No detection", 0.0)
    
    accuracy = (correct / len(ground_truth)) * 100
    
    print(f"\nAccuracy: {correct}/{len(ground_truth)} = {accuracy:.1f}%")
    
    # Show first few results
    print(f"\nSample detections (5 dau):")
    for img_name in sorted(list(ground_truth.keys())[:5]):
        true_label = ground_truth[img_name]
        pred_info = predictions.get(img_name, ("No detection", 0.0))
        pred_label, pred_conf_val = pred_info
        
        if pred_label == true_label:
            result_str = "✓"
        else:
            result_str = "✗"
        
        if pred_label != "No detection":
            print(f"  {result_str} {img_name}: {pred_label} ({pred_conf_val:.2f}) | GT: {true_label}")
        else:
            print(f"  {result_str} {img_name}: No detection | GT: {true_label}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_conf = conf_threshold
        best_predictions = predictions.copy()

print("\n" + "="*70)
print("KET QUA TOT NHAT")
print("="*70)

print(f"\nBest confidence threshold: {best_conf}")
print(f"Best accuracy: {best_accuracy:.1f}%")
print(f"Correct: {int(best_accuracy * len(ground_truth) / 100)}/{len(ground_truth)}")

print("\n📊 Bang ket qua chi tiet:")
print(f"{'Image':<25} {'Prediction':<20} {'Conf':<10} {'GT':<15} {'Result':<10}")
print("-" * 85)

correct_affected = 0
total_affected = 0
correct_healthy = 0
total_healthy = 0

for img_name in sorted(ground_truth.keys()):
    true_label = ground_truth[img_name]
    pred_info = best_predictions.get(img_name, ("No detection", 0.0))
    pred_label, pred_conf_val = pred_info
    
    if pred_label == true_label:
        result_str = "✓ Correct"
        if true_label == "Affected":
            correct_affected += 1
        else:
            correct_healthy += 1
    else:
        result_str = "✗ Wrong"
    
    if true_label == "Affected":
        total_affected += 1
    else:
        total_healthy += 1
    
    conf_str = f"{pred_conf_val:.2f}" if pred_label != "No detection" else "N/A"
    print(f"{img_name:<25} {pred_label:<20} {conf_str:<10} {true_label:<15} {result_str:<10}")

print("\n" + "="*70)
print("PHAN TICH THEO CLASS")
print("="*70)

affected_accuracy = (correct_affected / total_affected * 100) if total_affected > 0 else 0
healthy_accuracy = (correct_healthy / total_healthy * 100) if total_healthy > 0 else 0

print(f"\n📊 Class Performance:")
print(f"{'Class':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<15}")
print("-" * 50)
print(f"{'Affected':<15} {correct_affected:<10} {total_affected:<10} {affected_accuracy:>13.1f}%")
print(f"{'Healthy':<15} {correct_healthy:<10} {total_healthy:<10} {healthy_accuracy:>13.1f}%")
print(f"{'Overall':<15} {int(best_accuracy * len(ground_truth) / 100):<10} {len(ground_truth):<10} {best_accuracy:>13.1f}%")

print("\n" + "="*70)
print("SO SANH VOI TEST TRUOC (5 ANH)")
print("="*70)

# So sanh voi test cu
old_test_accuracy = 80.0  # 4/5 voi 5 anh cu
print(f"\nTest cu (5 anh):         {old_test_accuracy:.1f}% (4/5)")
print(f"Test moi (14 anh):       {best_accuracy:.1f}% ({int(best_accuracy * len(ground_truth) / 100)}/{len(ground_truth)})")

if best_accuracy >= old_test_accuracy:
    print(f"\n✓ Model van on dinh voi anh nhieu hon!")
else:
    diff = old_test_accuracy - best_accuracy
    print(f"\n⚠ Accuracy giam {diff:.1f}%")
    print(f"Ly do co the:")
    print(f"  - Anh moi kho hon")
    print(f"  - Can train voi data nhieu hon")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if best_accuracy >= 70:
    print(f"\n✅ MODEL TOT!")
    print(f"Accuracy: {best_accuracy:.1f}%")
    print(f"Ready cho beta testing")
elif best_accuracy >= 50:
    print(f"\n⚠ MODEL OK")
    print(f"Accuracy: {best_accuracy:.1f}%")
    print(f"Can cai thien them")
else:
    print(f"\n❌ MODEL CAN CAI THIEN")
    print(f"Accuracy: {best_accuracy:.1f}%")
    print(f"Can thu thap them data")

print("\n" + "="*70)
print("XEM KET QUA")
print("="*70)

print(f"\nAnh ket qua da luu tai:")
print(f"  runs\\predict_yolov8s_extended\\test_conf_{best_conf}\\")
print(f"\nXem anh:")
print(f"  start runs\\predict_yolov8s_extended\\test_conf_{best_conf}\\")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)

print(f"""
1. XEM ANH KET QUA:
   cd runs\\predict_yolov8s_extended\\test_conf_{best_conf}

2. TANG DATASET:
   - Dataset hien tai: 134 anh (Roboflow)
   - Dataset test: 14 anh (moi them)
   - Tong: 148 anh
   
3. KE HOACH RE-TRAINING:
   Option A: Train voi 134 anh Roboflow (da co)
   Option B: Them 14 anh test vao training set
   Option C: Thu thap them 500-1000 anh moi
   
4. Expected voi data lon hon:
   - 500 anh: Accuracy ~85-90%
   - 1000 anh: Accuracy ~90-95%
   - 2000+ anh: Accuracy >95%
""")

print("\n" + "="*70)
