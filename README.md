# StrawMind 🍄
**An Attention-Augmented Real-Time Detector and a Reproducible Imbalanced Benchmark for Mushroom Fungal-Disease Detection**

Detecting fungal contamination (*Trichoderma* green mold, *Aspergillus*) in mushroom cultivation,
with a transparent class-imbalance protocol and a CBAM-augmented YOLOv8 detector (**StrawMind-CBAM**).

## Key results (held-out TEST set, seed 42)
| Model | mAP@50 | mAP@50-95 | P | R | Params |
|---|---|---|---|---|---|
| YOLOv8n | 0.6937 | 0.5290 | 0.724 | 0.691 | 3.0M |
| YOLOv8s | 0.6970 | 0.5406 | 0.714 | 0.689 | 11.1M |
| YOLOv10n | 0.6873 | 0.5402 | 0.704 | 0.683 | 2.27M |
| **StrawMind-CBAM (ours)** | **0.6953** | 0.5385 | 0.712 | 0.684 | 3.10M |

Per-class mAP@50 (StrawMind-CBAM): Healthy **0.804** · Trichoderma **0.287** · Aspergillus **0.995**.
3-seed mean±std: see `RESULTS_3SEED.csv` (campaign in progress).

> ⚠️ **Every number is read directly from Ultralytics training/validation logs. No estimated or fabricated metrics.**

## Dataset (StrawMind v2)
- Aggregated from 2 public Roboflow Universe sources (CC BY 4.0), 3 classes (Healthy / Trichoderma / Aspergillus).
- Stratified 70/15/15 split (seed 42), **absolutely held-out test set**.
- Imbalance mitigation: Aspergillus oversampled ×4 (train only) + copy-paste augmentation.
- 1713 train / 276 val / 276 test images. See `DATASET_CARD.md`.

## StrawMind-CBAM architecture
YOLOv8n backbone + neck, with a **CBAM module on each detection scale (P3/P4/P5)** before the Detect head.
Warm-started from `yolov8n.pt`. Channels 64/128/256. See `configs/yolov8n-cbam.yaml`.

## Reproducibility
| Stage | Kaggle notebook |
|---|---|
| Dataset build | `cattillallnight/strawmind-dataprep` |
| Baselines YOLOv8n/s | `strawmind-lo1` |
| YOLOv10n + StrawMind-CBAM | `strawmind-lo2` / `strawmind-lo2b` |
| 3-seed campaign | `strawmind-lo3` |
| Ablation (CBAM × copy-paste) | `strawmind-lo4` |

All trained for 100 epochs, imgsz 640, batch 16, on NVIDIA P100, PyTorch 2.5.1 + CUDA 12.1, Ultralytics.

## Repo layout
```
paper.tex             # Q1 manuscript (LaTeX)
references.bib        # 11 DOI-verified citations
RELATED_WORK.md       # literature review draft
DATASET_CARD.md       # dataset documentation
RESULTS_TABLE_REAL.csv# verified per-model metrics
RESULTS_3SEED.csv     # 3-seed mean±std (in progress)
NOVELTY_STATEMENT.md  # contributions
configs/              # model + data yaml
notebooks/            # Kaggle training scripts
```

## License
Code: MIT. Data: per original sources (CC BY 4.0).
