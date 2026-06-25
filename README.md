# StrawMind: AIoT System for Straw Mushroom Disease Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-green.svg)](https://github.com/ultralytics/ultralytics)

StrawMind is a novel Edge AIoT system designed for real-time detection and management of fungal diseases in straw mushroom cultivation, specifically tailored for environments like the Mekong Delta in Vietnam.

## 🌟 Key Features

- **Lightweight Disease Detection**: Optimized StrawMind-YOLO model for real-time identification of Trichoderma spp., Aspergillus spp., and Soft Rot on edge devices (Raspberry Pi).
- **Comprehensive Dataset**: A curated and expanded dataset of straw mushroom diseases with meticulous annotations and advanced augmentations.
- **AIoT Integration**: Closed-loop environmental control based on real-time disease detection to proactively manage cultivation microclimates.
- **Scientific Rigor**: Developed following Q1 journal standards, including thorough audits, novelty statements, and SOTA baseline comparisons.

## 📁 Project Structure

```text
strawmind_final/
├── docs/               # Research and project documentation
│   ├── audit/          # Initial codebase audit and original reports
│   ├── research/       # Novelty statement, related work, and SOTA guidelines
│   ├── dataset/        # DATASET_CARD.md and data splitting strategy
│   ├── paper/          # LaTeX manuscript and references
│   └── QUALITY_CHECKLIST.md
├── src/                # Original and improved source code
│   ├── models/         # Model architectures (including CBAM)
│   ├── utils/          # Utility scripts for visualization and processing
│   └── inference/      # Demo and testing scripts
├── data/               # Dataset storage
│   ├── raw/            # Original training data and YAML configs
│   └── processed/      # Placeholder for augmented/standardized data
├── results/            # Experimental outputs
│   ├── tables/         # LaTeX results tables
│   ├── figures/        # Architecture diagrams and training plots
│   └── logs/           # Training logs
├── scripts/            # SOTA training and evaluation templates
├── assets/             # Pre-trained model weights (yolov8s_best.pt)
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/oonathuyquyen-collab/kaboomstrawmind.git
cd kaboomstrawmind
pip install -r src/requirements.txt
```

### 2. Inference Demo

Run the demo inference script using the pre-trained weights:

```bash
python src/demo_inference.py --weights assets/yolov8s_best.pt --source data/raw/test_images
```

### 3. Training Template

For establishing SOTA baselines, refer to the template script:

```bash
python scripts/training_script_template.py
```

## 📝 Research Artifacts

This project includes several key artifacts aimed at Q1 publication:

- **Audit Report**: Detailed analysis of the initial codebase and identified gaps.
- **Novelty Statement**: Articulation of the project's unique scientific contributions.
- **LaTeX Manuscript**: A professionally formatted paper draft ready for further refinement.
- **Dataset Card**: Comprehensive metadata and splitting strategy for the expanded dataset.

## 🤝 Contributing

Contributions to improve the model architecture, expand the dataset, or enhance the AIoT integration are welcome. Please refer to the research documentation in `docs/` for guidance on scientific standards.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Made with ❤️ for Vietnamese Mushroom Farmers 🍄
- Initial codebase by [cattillallnight](https://github.com/cattillallnight/strawmind)
- Research and refinement by Manus AI

---
*For more detailed information, please explore the `docs/` directory.*
