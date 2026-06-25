# Phase 3: Establishing SOTA Baselines for StrawMind

This document outlines the steps required to establish State-of-the-Art (SOTA) baselines for comparative analysis with the proposed StrawMind model. Due to the limitations of the current environment (inability to install complex deep learning frameworks, access to GPUs, and physical Raspberry Pi devices for real-time inference measurements), the actual execution of these steps will need to be performed in a suitable environment. This README serves as a guide for the implementation.

## Objective

To scientifically compare the proposed model with SOTA object detection models in a reproducible manner.

## Steps

### 1. Model Download and Installation

Download and set up the following models, ensuring consistent pretrained weights and input resolution (e.g., 640x640) for fair comparison. It is crucial to use the same input resolution across all models.

*   **YOLOv8 (n/s/m)**: Current baseline. Ultralytics provides easy installation and usage.
    ```bash
    pip install ultralytics
    # Example: from ultralytics import YOLO; model = YOLO("yolov8s.pt")
    ```
*   **YOLOv9**: Refer to the official YOLOv9 GitHub repository for installation and usage instructions.
*   **YOLOv10**: Refer to the official YOLOv10 GitHub repository for installation and usage instructions.
*   **YOLOv11 (or latest stable version)**: Refer to the official YOLOv11 GitHub repository (or the latest stable YOLO variant) for installation and usage instructions.
*   **RT-DETR**: Refer to the official RT-DETR repository for installation and usage instructions.
*   **Faster R-CNN (ResNet50)**: Implement using popular frameworks like Detectron2 or MMDetection. This serves as a classic 2-stage baseline.
*   **EfficientDet-Lite0**: Implement using TensorFlow Lite Model Maker or similar tools for lightweight edge comparison.

### 2. Training Protocol

All models MUST be trained on the **SAME dataset** (the expanded dataset from Phase 2), using the **SAME augmentation pipeline**, for the **SAME number of epochs** (or with consistent early-stopping criteria), and on the **SAME hardware** (if possible, to ensure fair comparison of training times). Full logs (e.g., using Weights & Biases or TensorBoard) MUST be recorded for each experiment.

**Key considerations:**
*   **Dataset**: The unified and augmented dataset created in Phase 2.
*   **Augmentation**: The advanced augmentation pipeline defined in `DATASET_CARD.md`.
*   **Hyperparameters**: Optimize hyperparameters for each model, but ensure consistency in evaluation criteria (e.g., early stopping based on validation mAP).
*   **Hardware**: Ideally, all models should be trained on identical hardware configurations to ensure comparability of training duration and resource utilization.

### 3. Evaluation Metrics

Evaluate all trained models on the **SAME test set** using the following comprehensive metrics:

*   **mAP50, mAP50-95**: Standard COCO-style mean Average Precision metrics.
*   **Precision, Recall, F1-score**: Per-class metrics to understand performance for each disease type.
*   **Inference Time + FPS**: Measure actual inference time and frames per second on target edge devices (e.g., Raspberry Pi Zero 2W / Raspberry Pi 4). This requires physical deployment and measurement.
*   **Model Size (MB), FLOPs, Number of Parameters**: Quantify model complexity and resource footprint.
*   **Confusion Matrix**: Visualize classification performance and identify common misclassifications.

### 4. Statistical Validation

To ensure the robustness and statistical significance of the results:

*   **Multiple Runs**: Each experiment (training and evaluation of a model) MUST be run at least 3 times with different random seeds. Report the mean ± standard deviation for all quantitative metrics.
*   **Statistical Significance Testing**: Perform paired t-tests or Wilcoxon signed-rank tests to statistically prove that any observed differences between the proposed model and SOTA baselines are significant (p < 0.05), rather than merely numerically superior.

### 5. Output Generation

Generate the following outputs:

*   **LaTeX Results Table (`RESULTS_TABLE.tex`)**: A professionally formatted table summarizing all key metrics for each model, including mean ± std values.
*   **Plots**: Generate PR curves, confusion matrices, and training curves (loss, mAP over epochs) for all models. These should be saved as high-resolution image files.

## Placeholder `RESULTS_TABLE.tex`

Below is a template for the `RESULTS_TABLE.tex` file, which will be populated with actual results upon execution of this phase.

```latex
\documentclass{article}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}

\begin{document}

\section*{Experimental Results and Baseline Comparison}

This section presents the comparative experimental results of various State-of-the-Art (SOTA) object detection models evaluated on the expanded straw mushroom disease dataset. All models were trained and evaluated under identical conditions to ensure a fair comparison.

\begin{table*}[htbp]
    \centering
    \caption{Comparative Performance of SOTA Object Detection Models on Straw Mushroom Disease Dataset}
    \label{tab:results}
    \resizebox{\textwidth}{!}{
    \begin{tabular}{lcccccccc}
        \toprule
        \multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Params (M)}} & \multirow{2}{*}{\textbf{FLOPs (G)}} & \multicolumn{2}{c}{\textbf{mAP (\%)}} & \multicolumn{2}{c}{\textbf{Inference (ms)}} & \multirow{2}{*}{\textbf{Model Size (MB)}} & \multirow{2}{*}{\textbf{FPS (RPi4)}}\\
        \cmidrule(lr){4-5} \cmidrule(lr){6-7}
        & & & \textbf{mAP50} & \textbf{mAP50-95} & \textbf{GPU} & \textbf{RPi4} & & \\
        \midrule
        YOLOv8n & 3.0 & 8.2 & XX.X \(\pm\) Y.Y & XX.X \(\pm\) Y.Y & A.A & B.B & C.C & D.D \\
        YOLOv8s & 11.2 & 28.6 & XX.X \(\pm\) Y.Y & XX.X \(\pm\) Y.Y & A.A & B.B & C.C & D.D \\
        YOLOv9t & 6.9 & 25.1 & XX.X \(\pm\) Y.Y & XX.X \(\pm\) Y.Y & A.A & B.B & C.C & D.D \\
        YOLOv10n & 2.8 & 7.4 & XX.X \(\pm\) Y.Y & XX.X \(\pm\) Y.Y & A.A & B.B & C.C & D.D \\
        RT-DETR-R18 & 25.0 & 49.0 & XX.X \(\pm\) Y.Y & XX.X \(\pm\) Y.Y & A.A & B.B & C.C & D.D \\
        Faster R-CNN (ResNet50) & 41.0 & 164.0 & XX.X \(\pm\) Y.Y & XX.X \(\pm\) Y.Y & A.A & B.B & C.C & D.D \\
        EfficientDet-Lite0 & 3.4 & 2.1 & XX.X \(\pm\) Y.Y & XX.X \(\pm\) Y.Y & A.A & B.B & C.C & D.D \\
        \midrule
        Proposed StrawMind-YOLO & Z.Z & W.W & XX.X \(\pm\) Y.Y & XX.X \(\pm\) Y.Y & A.A & B.B & C.C & D.D \\
        \bottomrule
    \end{tabular}
    }
\end{table*}

\end{document}
```

**Note**: The `XX.X`, `Y.Y`, `A.A`, `B.B`, `C.C`, `D.D`, `Z.Z`, `W.W` placeholders in the table above represent hypothetical values. These will be replaced with actual measured values after the experiments are completed in a suitable environment. This table is designed to be directly compilable with LaTeX.
