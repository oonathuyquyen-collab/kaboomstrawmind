# NOVELTY_STATEMENT — StrawMind

> Lưu ý: đây là tuyên bố đóng góp dựa trên thiết kế hệ thống. Mọi con số hiệu năng
> chỉ được điền sau khi có kết quả thật (RULE 3/4). Related Work + citation sẽ bổ sung
> sau khi search literature thật (Semantic Scholar / Scholar), KHÔNG bịa citation.

## Bối cảnh
Phát hiện bệnh nấm (Trichoderma green mold, Aspergillus) ở quy mô trang trại bằng
object detection thời gian thực, hướng triển khai edge (Raspberry Pi-class).

## Contributions (claim, cần kết quả thật để chứng minh)
1. **Unified 3-class mushroom-disease detection dataset** gộp từ 2 nguồn Roboflow công khai
   (disease-mushroom, ma3), remap về {Healthy, Trichoderma, Aspergillus}, chia
   stratified 70/15/15 với **test set held-out tuyệt đối**. Công khai số đếm thật per-class
   (DATASET_CARD.md). → Khắc phục tình trạng dataset cũ chỉ 134 ảnh, nhãn không khớp.
2. **Xử lý mất cân bằng lớp nghiêm trọng (~29×)** cho Aspergillus bằng kết hợp
   oversampling có kiểm soát (chỉ train split) + copy_paste augmentation; báo cáo
   minh bạch như một limitation.
3. **So sánh có kiểm soát (controlled benchmark)** giữa các kiến trúc YOLO
   (YOLOv8n/s, YOLOv10n) và biến thể đề xuất **StrawMind-CBAM** (YOLOv8 + CBAM attention),
   cùng dataset, cùng seed, cùng augmentation → kết luận công bằng.
4. **Ablation study** tách biệt đóng góp của từng thành phần (attention, augmentation
   imbalance-handling, backbone) — chỉ kết luận khi có số thật, đa seed.
5. **Reproducibility-first**: toàn bộ pipeline chạy trên Kaggle Notebook công khai
   (dataprep + train), seed cố định, log thật được trích xuất bằng code (không nhập tay).

## NOVELTY chính (1 câu)
StrawMind là một **benchmark + pipeline reproducible cho phát hiện bệnh nấm 3 lớp
mất cân bằng**, trong đó biến thể StrawMind-CBAM tích hợp attention nhằm cải thiện
phát hiện lớp hiếm (Aspergillus) — với toàn bộ số liệu được xác minh từ log thật.

## Điều KHÔNG claim (để tránh reviewer bác)
- KHÔNG claim đo tốc độ inference trên Raspberry Pi nếu chưa đo trên phần cứng thật.
- KHÔNG claim SOTA tuyệt đối; chỉ claim so sánh công bằng trong phạm vi dataset này.
- KHÔNG dùng 134 ảnh nấm rơm gốc để tính mAP (nhãn sai class) — chỉ qualitative.
