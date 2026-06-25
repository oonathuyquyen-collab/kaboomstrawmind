# Related Work Table

| Tác giả | Năm | Phương pháp | Dataset Size | Kết quả chính | Hạn chế |
|---|---|---|---|---|---|
| Choi et al. [1] | 2026 | YOLOv8-seg, YOLOv11-seg | Không rõ (nấm trồng trong điều kiện kiểm soát) | YOLOv11-seg có hiệu suất tương đương YOLOv8-seg trong phát hiện và phân đoạn nấm ăn được. mAP50-95 ~0.55. | Dataset nhỏ, điều kiện kiểm soát, không đánh giá trên môi trường thực tế đa dạng. |
| Dika & Iqbal [2] | 2026 | CNN | 1,200 ảnh nấm (khỏe mạnh/bệnh) | Độ chính xác tổng thể 92.5% trong phát hiện bệnh nấm. | Chỉ phân loại khỏe mạnh/bệnh, không phân loại loại bệnh cụ thể. Dataset nhỏ, không đánh giá trên thiết bị biên. |
| YOLO-AREL (ScienceDirect) [3] | 2026 | YOLO-AREL (dựa trên YOLOv11, tích hợp Adown module, RepViTBlock + EMA, LQEHead) | 4,840 ảnh (12 loại nấm, bao gồm nấm độc) | Cải thiện hiệu suất phát hiện nấm dại trong môi trường phức tạp. Khả thi cho phát hiện thời gian thực trên RTX 4060. | Tập trung vào nấm dại, không chuyên biệt cho bệnh nấm rơm. Đánh giá trên GPU mạnh, chưa tối ưu cho thiết bị biên tài nguyên hạn chế. |

## References

[1] Choi, D.-H.; Oh, Y.-L.; Oh, M.; Lee, E.-J.; Woo, S.-I.; Kim, M.; Im, J.-H. Comparative Evaluation of YOLOv8 and YOLOv11 for Digital Phenotyping of Edible Mushrooms Under Controlled Cultivation Conditions. *J. Fungi* **2026**, *12*, 232. Available online: [https://www.mdpi.com/2309-608X/12/4/232](https://www.mdpi.com/2309-608X/12/4/232)
[2] Dika, D.; Iqbal, M. Fungal Disease Detection Using CNN Deep Learning Method. *JADEN Journal of Algorithmic Digital Engineering and Networks* **2026**, *1*, 42-48. Available online: [https://ejournal.nascade.org/index.php/jaden/article/view/120](https://ejournal.nascade.org/index.php/jaden/article/view/120)
[3] YOLO-AREL: A lightweight algorithm for wild mushroom species recognition in complex environments. *ScienceDirect* **2026**. Available online: [https://www.sciencedirect.com/science/article/pii/S2772375526001437](https://www.sciencedirect.com/science/article/pii/S2772375526001437)
