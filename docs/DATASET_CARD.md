# DATASET_CARD — StrawMind v2 (70/15/15 held-out test)

## Nguồn (Roboflow Universe, CC BY 4.0)
- Source A: fungi-buster-1hvwz/disease-mushroom — giữ Healthy, Trichoderma (bỏ Bacterial Blotch, Dry Bubble, Wilt)
- Source B: minor-a4s64/ma3 (v4) — giữ Trichoderma, aspergillus, healthy → cả 3 lớp
- Remap: 0=Healthy, 1=Trichoderma, 2=Aspergillus (match không phân biệt hoa thường; "green mold"→Trichoderma)

## Split: stratified 70/15/15 (seed=42), TEST là held-out tuyệt đối (không động tới khi train)
| Class | Train img | Train bbox | Val img | Val bbox | Test img | Test bbox |
|---|---|---|---|---|---|---|
| Healthy     | 603 | 779  | 129 | 191 | 129 | 184 |
| Trichoderma | 546 | 3948 | 117 | 859 | 117 | 981 |
| Aspergillus | 564*| 564  | 30  | 30  | 30  | 30  |

\* Aspergillus train ĐÃ oversample ×4 (141→564 ảnh) để giảm imbalance.

## Xử lý mất cân bằng (Aspergillus hiếm)
1. **Oversampling**: nhân bản ảnh Aspergillus trong TRAIN ×4 (chỉ train, val/test giữ nguyên).
2. **copy_paste=0.5** augmentation khi train.
- Ghi chú: Ultralytics KHÔNG hỗ trợ per-class cls-weight trực tiếp → dùng oversampling làm phương án tương đương (reviewer chấp nhận).

## Limitations (đưa vào paper)
- Imbalance gốc ~29× (Trichoderma 4728 vs Aspergillus 165 bbox toàn bộ) → recall Aspergillus có thể thấp.
- Test set ban đầu (134 ảnh nấm rơm, nhãn Affected/Healthy/Healthy-Affected) KHÔNG dùng tính mAP (sai class) — chỉ giữ làm "additional qualitative visualization on straw mushroom images".

## Reproducibility
- Notebook: cattillallnight/strawmind-dataprep (seed=42, internet on). Output dùng trực tiếp qua kernel_sources.
