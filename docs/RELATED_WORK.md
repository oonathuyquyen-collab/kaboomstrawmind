# RELATED_WORK (draft) — StrawMind

> Tất cả citation dưới đây là **paper thật, DOI verified qua Crossref**. Mục "cần verify"
> được tách riêng, KHÔNG đưa vào paper cho tới khi xác minh thủ công (RULE: không bịa citation).

## 1. Real-time object detection (nền tảng)
Object detection thời gian thực khởi nguồn từ YOLO [redmon2016yolo], đặt nền cho dòng
one-stage detector. Phiên bản mới nhất YOLOv10 [wang2024yolov10] loại bỏ NMS và tối ưu
end-to-end cho tốc độ/độ chính xác, với nhiều cải tiến cho môi trường phức tạp
[wu2025improvedyolov10]. StrawMind dùng họ YOLO (v8n/s, v10n) làm baseline có kiểm soát.

## 2. Attention cho phát hiện bệnh nông nghiệp
Cơ chế attention cải thiện nhận diện bệnh cây trồng: nhận dạng bệnh lúa với attention +
domain adaptation trên chính tạp chí mục tiêu *Computers and Electronics in Agriculture*
[chen2023domainrice]; phát hiện bệnh chanh dây bằng sparse parallel attention [he2025passionfruit].
StrawMind-CBAM bổ sung CBAM attention trên từng scale phát hiện để tăng độ nhạy với lớp hiếm.
(*CBAM gốc Woo et al. 2018 — cần verify DOI trước khi cite*.)

## 3. Mất cân bằng lớp (class imbalance / long-tailed)
Focal Loss [lin2017focal] và biến thể long-tailed Equalized Focal Loss [li2022equalized]
là hướng chuẩn xử lý imbalance trong detection. Do Ultralytics không hỗ trợ per-class
loss-weight trực tiếp, StrawMind dùng **oversampling có kiểm soát** kết hợp augmentation.

## 4. Data augmentation (copy-paste)
Copy-paste augmentation tăng đa dạng mẫu và hỗ trợ object hiếm/nhỏ
[zhang2023copypaste]. StrawMind đặt copy_paste=0.5 cho lớp Aspergillus hiếm.

## 5. Khoảng trống (research gap)
Chưa có benchmark reproducible, công khai, **3 lớp bệnh nấm mất cân bằng**
(Healthy/Trichoderma/Aspergillus) với test set held-out đúng class + xử lý imbalance minh bạch.
StrawMind lấp khoảng trống này.

---
### Cần verify thủ công trước khi đưa vào paper
- Woo et al. (2018) "CBAM: Convolutional Block Attention Module", ECCV — nguồn gốc CBAM.
- Jocher et al. (2023) Ultralytics YOLOv8 — phần mềm, cite qua repo (không DOI).
