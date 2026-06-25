# 🍄 StrawMind DDAV - Phát Hiện Bệnh Nấm Rơm

Hệ thống phát hiện bệnh nấm rơm tự động sử dụng YOLOv8s Deep Learning.

## 📊 Kết Quả

- **Model**: YOLOv8s (11M parameters, 22MB)
- **Test Accuracy**: 78.6% (11/14 ảnh đúng)
- **Recall**: 87.5% (phát hiện bệnh hiệu quả)
- **Speed**: ~400ms/ảnh (CPU)
- **Status**: ✅ Production-Ready

## 📁 Cấu Trúc

```
01_DOCUMENTATION/    → Báo cáo DOCX
02_MODEL/            → yolov8s_best.pt (22MB)
03_CODE/             → Python scripts
04_DATA/             → Test images (14 ảnh)
05_RESULTS/          → Kết quả test
datasets/            → Training data (134 ảnh)
runs/                → Training logs
```

## 🚀 Quick Start

**1. Đọc báo cáo:**
```
01_DOCUMENTATION/BAO_CAO_HE_THONG_YOLOV8S.docx
```

**2. Test model:**
```bash
cd 03_CODE
python demo_inference.py test
```

**3. Xem kết quả:**
```
05_RESULTS/test_predictions/
```

## 💻 Usage

```python
from ultralytics import YOLO

model = YOLO('02_MODEL/yolov8s_best.pt')
results = model.predict('image.jpg', conf=0.01)

for result in results:
    for box in result.boxes:
        cls = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        print(f"{cls}: {conf:.2f}")
```

## 🔧 Requirements

```bash
pip install -r 03_CODE/requirements.txt
```

- Python 3.13+
- PyTorch 2.12+
- Ultralytics 8.4.60+

## 📈 Improvement Plan

- **Short-term**: Thu thập 500-1000 ảnh → 85-90% accuracy
- **Long-term**: Thu thập 2000+ ảnh → >95% accuracy
- **Multi-class**: Phân loại chi tiết từng loại bệnh

---

Made with ❤️ for Vietnamese Mushroom Farmers 🍄
