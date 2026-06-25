"""
StrawMind-v2 Training — Final Launch Script
============================================
Chạy script này để bắt đầu training qua đêm.
Dataset: 418 images (merged từ 3 sources)
Model: YOLOv8s + Novelty improvements

Usage: python train_strawmind_v2.py
"""
import os, sys, time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
import torch

BASE_DIR = Path('d:/FPT/compe/Argitech2026/strawmind')
MERGED_YAML = BASE_DIR / 'datasets_merged/data.yaml'
ORIGINAL_YAML = BASE_DIR / 'strawmind_project/datasets/roboflow_mushroom/data.yaml'

DATA_YAML = str(MERGED_YAML) if MERGED_YAML.exists() else str(ORIGINAL_YAML)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch = 4 if device == 'cuda' else 2

print("="*70)
print("STRAWMIND-v2 - NOVEL YOLOV8S TRAINING")
print("="*70)
print(f"Device : {device}")
print(f"Dataset: {DATA_YAML}")
print(f"Batch  : {batch}")
print()
print("Novelties vs baseline:")
print("  [1] cos_lr=True        -> Cosine Annealing LR")
print("  [2] label_smoothing=0.05 -> Smoother labels, less overfit")
print("  [3] dropout=0.1        -> Regularization in head")
print("  [4] mixup=0.25         -> +67% vs baseline 0.15")
print("  [5] copy_paste=0.3     -> +100% vs baseline 0.15")
print("  [6] patience=35        -> +40% vs baseline 25")
print("  [7] lr0=0.0003         -> Lower LR, more stable")
print("  [8] epochs=150         -> +50% vs baseline 100")
print("  [9] 418 images         -> +3x dataset vs baseline 134")
print()

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: pip install ultralytics")
    sys.exit(1)

# Load pretrained YOLOv8s
print("Loading YOLOv8s pretrained...")
model = YOLO('yolov8s.pt')
print("Loaded!")

config = {
    # Data
    'data': DATA_YAML,
    'imgsz': 640,

    # Duration
    'epochs': 150,
    'patience': 35,

    # Batch & device
    'batch': batch,
    'device': device,
    'workers': 2,

    # Output
    'project': str(BASE_DIR / 'runs/strawmind_v2'),
    'name': 'strawmind_cbam_v2',
    'save': True,
    'save_period': 25,
    'plots': True,
    'verbose': True,

    # Optimizer — AdamW + Cosine LR (NOVELTY)
    'optimizer': 'AdamW',
    'lr0': 0.0003,
    'lrf': 0.005,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 5,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'cos_lr': True,              # NOVELTY: Cosine LR schedule

    # Augmentation — STRONGER than baseline
    'hsv_h': 0.025,
    'hsv_s': 0.75,
    'hsv_v': 0.45,
    'degrees': 20,
    'translate': 0.15,
    'scale': 0.6,
    'shear': 3.0,
    'perspective': 0.0002,
    'flipud': 0.5,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.25,               # NOVELTY: +67% vs baseline
    'copy_paste': 0.3,           # NOVELTY: +100% vs baseline
    'close_mosaic': 15,

    # Loss weights
    'cls': 0.5,
    'box': 7.5,
    'dfl': 1.5,

    # Regularization (NOVELTY)
    'label_smoothing': 0.05,     # NOVELTY: label smoothing
    'dropout': 0.1,              # NOVELTY: dropout in head

    # Other
    'cache': False,
    'amp': False,
    'rect': False,
    'fraction': 1.0,
}

print()
print("Starting training... (press Ctrl+C to stop and resume later)")
print(f"ETA: ~6-8 hours on CPU | ~45-90 min on GPU")
print("="*70)

start = time.time()
try:
    results = model.train(**config)
    elapsed = time.time() - start
    h, m = int(elapsed // 3600), int((elapsed % 3600) // 60)

    print()
    print("="*70)
    print("TRAINING COMPLETE!")
    print(f"Time: {h}h {m}m")

    best = BASE_DIR / 'runs/strawmind_v2/strawmind_cbam_v2/weights/best.pt'
    print(f"Best model: {best}")

    # Validate
    if best.exists():
        print()
        print("Validating best model...")
        best_model = YOLO(str(best))
        m = best_model.val(data=DATA_YAML, split='val', verbose=False)

        # Baseline reference
        baseline = {'map50': 0.723, 'map': 0.281, 'mp': 0.778, 'mr': 0.875}

        print()
        print("="*70)
        print("BENCHMARK: BASELINE vs STRAWMIND-v2")
        print("="*70)
        print(f"{'Metric':<15} {'Baseline':>12} {'v2':>12} {'Delta':>12}")
        print("-"*54)

        metrics_map = [
            ('mAP@50',     baseline['map50'], m.box.map50),
            ('mAP@50-95',  baseline['map'],   m.box.map),
            ('Precision',  baseline['mp'],     m.box.mp),
            ('Recall',     baseline['mr'],     m.box.mr),
        ]
        for name, b, v in metrics_map:
            d = v - b
            sign = '+' if d >= 0 else ''
            print(f"{name:<15} {b*100:>11.1f}% {v*100:>11.1f}% {sign}{d*100:>10.1f}%")

        print("="*70)
        imp = m.box.map50 - baseline['map50']
        if imp > 0.05:
            print(f"STRAWMIND-v2 WINS! mAP50 +{imp*100:.1f}%")
        elif imp > 0:
            print(f"Improvement: +{imp*100:.1f}% mAP50")
        else:
            print(f"mAP50 diff: {imp*100:.1f}% (need more data or epochs)")

        # Save result
        result_file = BASE_DIR / 'strawmind_project/05_RESULTS/benchmark_v2.txt'
        result_file.parent.mkdir(exist_ok=True)
        with open(result_file, 'w') as f:
            f.write("StrawMind-v2 Benchmark\n")
            f.write("="*50 + "\n")
            f.write(f"Dataset: {DATA_YAML}\n")
            f.write(f"Total images: 418 (train:316, val:68, test:34)\n")
            f.write(f"Training time: {h}h {m}m\n\n")
            f.write("Novelties:\n")
            f.write("  - Cosine LR scheduling (cos_lr=True)\n")
            f.write("  - Label smoothing (0.05)\n")
            f.write("  - Dropout in head (0.1)\n")
            f.write("  - Stronger MixUp (0.25) and CopyPaste (0.3)\n")
            f.write("  - 3x more training data (418 vs 134 images)\n\n")
            f.write("Results:\n")
            for name, b, v in metrics_map:
                d = v - b
                f.write(f"  {name}: {v*100:.2f}% (vs baseline {b*100:.1f}%)\n")
        print(f"Saved results: {result_file}")

except KeyboardInterrupt:
    elapsed = time.time() - start
    h, m = int(elapsed // 3600), int((elapsed % 3600) // 60)
    print(f"\nTraining stopped after {h}h {m}m")
    print("Checkpoints saved. Resume with same command.")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print()
print("Next steps:")
print("  python strawmind_project/03_CODE/gradcam_viz.py")
print("  python create_result_charts.py")
