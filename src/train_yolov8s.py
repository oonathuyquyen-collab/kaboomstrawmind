"""
Training voi YOLOv8s - Model LON HON
- YOLOv8n: 3M params
- YOLOv8s: 11M params (3.6x lon hon)
- Model size: ~22MB
- Expected: Accuracy cao hon nhung cham hon
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import torch
from pathlib import Path

print("="*70)
print("STRAWMIND DDAV - TRAINING VOI YOLOV8S (11M PARAMS)")
print("="*70)

# Check GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Su dung CPU")
    print("Canh bao: YOLOv8s lon hon, se cham hon tren CPU!")
    print("Thoi gian uoc tinh: ~3-4 gio (100 epochs)")

dataset_path = "datasets/roboflow_mushroom"

print("\n" + "="*70)
print("BUOC 1: SO SANH MODELS")
print("="*70)

print("\nYOLOv8n (dang su dung):")
print("  Parameters: 3.0M")
print("  Size: ~6MB")
print("  Speed: Fast")
print("  Accuracy: Medium")

print("\nYOLOv8s (se dung):")
print("  Parameters: 11.1M")
print("  Size: ~22MB")
print("  Speed: Medium")
print("  Accuracy: Higher")
print("  Trade-off: 3.6x lon hon, hy vong accuracy cao hon!")

print("\n" + "="*70)
print("BUOC 2: TAI YOLOV8S PRETRAINED")
print("="*70)

print("\nTai YOLOv8s pretrained weights...")
print("(Lan dau tai se download ~22MB)")
model = YOLO('yolov8s.pt')
print("✓ YOLOv8s loaded!")

print(f"\nModel info:")
print(f"  Architecture: YOLOv8s")
print(f"  Parameters: ~11M")
print(f"  Pretrained: COCO dataset")

print("\n" + "="*70)
print("BUOC 3: CAU HINH TRAINING")
print("="*70)

# Cau hinh giong model improved nhung voi YOLOv8s
config = {
    # Data
    'data': f'{dataset_path}/data.yaml',
    
    # Training duration
    'epochs': 100,
    'patience': 25,  # Tang patience vi model lon hon
    
    # Image & batch
    'imgsz': 640,
    'batch': 4 if device == 'cuda' else 2,  # Giam batch vi model lon
    'device': device,
    
    # Output
    'project': 'runs/train_yolov8s',
    'name': 'strawmind_yolov8s',
    'save': True,
    'plots': True,
    'verbose': True,
    
    # Optimization - TUNED cho model lon
    'optimizer': 'AdamW',
    'lr0': 0.0005,  # LR thap hon vi model lon
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 5,  # Tang warmup vi model lon
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    
    # Data Augmentation - MANH
    'hsv_h': 0.02,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 15,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 2.0,
    'perspective': 0.0,
    'flipud': 0.5,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.15,  # Tang mixup
    'copy_paste': 0.15,  # Tang copy-paste
    
    # Other
    'workers': 4,
    'cache': False,
    'amp': False,
    'close_mosaic': 10,
}

print("\nTraining configuration:")
print(f"\n1. MODEL:")
print(f"  YOLOv8s: 11M params (3.6x lon hon YOLOv8n)")
print(f"  Batch size: {config['batch']} (giam vi model lon)")

print(f"\n2. OPTIMIZATION:")
print(f"  Learning rate: {config['lr0']} (thap hon vi model lon)")
print(f"  Warmup: {config['warmup_epochs']} epochs (nhieu hon)")
print(f"  Patience: {config['patience']} (nhieu hon)")

print(f"\n3. AUGMENTATION:")
print(f"  Mixup: {config['mixup']} (tang)")
print(f"  Copy-paste: {config['copy_paste']} (tang)")

print("\n" + "="*70)
print("BUOC 4: TRAINING")
print("="*70)

print("\nBat dau training YOLOv8s...")
print("Thoi gian uoc tinh:")
if device == 'cuda':
    print("  GPU: ~1.5-2 gio")
else:
    print("  CPU: ~3-4 gio (LON HON 2x so voi YOLOv8n)")
print("\nMuc tieu: mAP50 > 20%, Accuracy > 50%\n")

try:
    results = model.train(**config)
    
    print("\n" + "="*70)
    print("TRAINING HOAN THANH!")
    print("="*70)
    
    print(f"\nModel weights:")
    print(f"  Best: runs/train_yolov8s/strawmind_yolov8s/weights/best.pt")
    print(f"  Size: ~22MB")
    
except KeyboardInterrupt:
    print("\n\nTraining bi dung")
except Exception as e:
    print(f"\n\nLoi: {e}")

print("\n" + "="*70)
print("BUOC 5: VALIDATE & SO SANH")
print("="*70)

try:
    best_model_path = "runs/detect/runs/train_yolov8s/strawmind_yolov8s/weights/best.pt"
    if os.path.exists(best_model_path):
        print(f"\nLoad best model: {best_model_path}")
        best_model = YOLO(best_model_path)
        
        print("\nValidate tren test set...")
        metrics = best_model.val(data=f'{dataset_path}/data.yaml', split='test')
        
        print("\n" + "="*70)
        print("YOLOV8S FINAL METRICS")
        print("="*70)
        
        print(f"\n  mAP50:     {metrics.box.map50:.3f} ({metrics.box.map50*100:.1f}%)")
        print(f"  mAP50-95:  {metrics.box.map:.3f} ({metrics.box.map*100:.1f}%)")
        print(f"  Precision: {metrics.box.mp:.3f} ({metrics.box.mp*100:.1f}%)")
        print(f"  Recall:    {metrics.box.mr:.3f} ({metrics.box.mr*100:.1f}%)")
        
        print("\n" + "="*70)
        print("SO SANH VOI CAC MODELS TRUOC")
        print("="*70)
        
        # So sanh
        yolov8n_old = {'map50': 0.0541, 'map': 0.0165, 'acc': 0.0}
        yolov8n_improved = {'map50': 0.181, 'map': 0.065, 'acc': 0.40}
        yolov8s_new = {'map50': metrics.box.map50, 'map': metrics.box.map}
        
        print(f"\n{'Model':<20} {'mAP50':<12} {'mAP50-95':<12} {'Test Acc':<12}")
        print("-" * 60)
        print(f"{'YOLOv8n (old)':<20} {yolov8n_old['map50']*100:>10.1f}% {yolov8n_old['map']*100:>10.1f}% {yolov8n_old['acc']*100:>10.1f}%")
        print(f"{'YOLOv8n (improved)':<20} {yolov8n_improved['map50']*100:>10.1f}% {yolov8n_improved['map']*100:>10.1f}% {yolov8n_improved['acc']*100:>10.1f}%")
        print(f"{'YOLOv8s (new)':<20} {yolov8s_new['map50']*100:>10.1f}% {yolov8s_new['map']*100:>10.1f}% {'?':>10}%")
        
        print("\n" + "="*70)
        print("DANH GIA")
        print("="*70)
        
        if yolov8s_new['map50'] > yolov8n_improved['map50']:
            improvement = ((yolov8s_new['map50'] - yolov8n_improved['map50']) / yolov8n_improved['map50']) * 100
            print(f"\n✓✓ YOLOV8S TOT HON!")
            print(f"mAP50 tang them: +{improvement:.1f}%")
            print(f"Model lon hon (11M vs 3M) giup accuracy cao hon!")
        elif yolov8s_new['map50'] > yolov8n_old['map50']:
            print(f"\n✓ YOLOv8s tot hon baseline")
            print(f"Nhung chua vuot YOLOv8n improved")
        else:
            print(f"\n⚠ YOLOv8s chua tot hon")
            print(f"Co the do dataset qua nho cho model lon")
            
except Exception as e:
    print(f"\nKhong the validate: {e}")

print("\n" + "="*70)
print("HOAN THANH!")
print("="*70)

print("""
Buoc tiep theo:
1. Test voi 5 anh: python test_yolov8s.py
2. So sanh 3 models: YOLOv8n old, YOLOv8n improved, YOLOv8s
3. Chon model tot nhat

Ket luan:
- Neu YOLOv8s > YOLOv8n improved: Dung YOLOv8s
- Neu YOLOv8s = YOLOv8n improved: Dung YOLOv8n (nhe hon)
- Model tot nhat se duoc dung cho production
""")
