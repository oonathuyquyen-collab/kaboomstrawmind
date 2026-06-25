# DATASET CARD for StrawMind Project

## 1. Dataset Sources and Details

To expand the existing limited dataset and address the ambiguity in class definitions, we propose integrating and curating images from the following publicly available datasets. The target classes for our expanded dataset will be: **Healthy**, **Mốc xanh (Trichoderma spp.)**, **Mốc đen (Aspergillus spp.)**, and **Thối nhũn (Soft Rot)**.

| Tên Dataset | Link | License | Số ảnh khả dụng (ước tính) | Ghi chú | Cần Permission/Citation? |
|---|---|---|---|---|---|
| **Mushroom Disease Dataset (Healthy, Single Infected & Mixed Infected)** | [https://data.mendeley.com/datasets/jrbx34k77g](https://data.mendeley.com/datasets/jrbx34k77g) | CC BY 4.0 | 761 (299 Healthy, 147 Single-infected (Green/Black mold), 315 Mixed-infected) | Rất phù hợp cho Trichoderma (Green mold) và Aspergillus (Black mold). Cần phân loại lại các ảnh "Mixed-infected" và "Single-infected" thành các class bệnh cụ thể. | Có, theo CC BY 4.0 |
| **Mushroom Disease and Species Multi Transformation** | [https://www.kaggle.com/datasets/shuvokumarbasak2030/mushroom-disease-and-species-multi-transformation](https://www.kaggle.com/datasets/shuvokumarbasak2030/mushroom-disease-and-species-multi-transformation) | MIT | 680 (299 Healthy, 72 Single-infected (Green/Black mold), 309 Mixed-infected) | Tương tự dataset trên, cung cấp thêm ảnh cho Trichoderma và Aspergillus. Cần phân loại lại. | Có, theo MIT |
| **Mushroom-Diseases-Dataset (Roboflow)** | [https://universe.roboflow.com/abdullah-hzljd/mushroom-diseases-dataset](https://universe.roboflow.com/abdullah-hzljd/mushroom-diseases-dataset) | CC BY 4.0 | Không rõ số lượng cụ thể, nhưng có các class Healthy, Black mold, Green mold. | Cần kiểm tra số lượng và chất lượng ảnh. | Có, theo CC BY 4.0 |
| **Cauliflower Dataset** | [https://data.mendeley.com/datasets/x26px3xnmy](https://data.mendeley.com/datasets/x26px3xnmy) | Không rõ (cần kiểm tra) | Chứa ảnh "Bacterial Soft Rot" | Có thể dùng cho class "Thối nhũn (Soft Rot)" nếu bệnh lý tương đồng và có thể chuyển giao. Cần kiểm tra license và tính phù hợp. | Có thể cần |
| **Existing StrawMind Dataset** | `/home/ubuntu/strawmind/datasets/roboflow_mushroom` | CC BY 4.0 | 134 (117 Train, 11 Valid, 6 Test) | Cần phân loại lại các nhãn "Affected", "Healthy-Affected" thành các class bệnh cụ thể hoặc loại bỏ nếu không rõ ràng. | Có, theo CC BY 4.0 |

## 2. Advanced Augmentation Pipeline Design

Beyond basic geometric transformations (flip, rotate), the following advanced augmentation techniques will be employed to enhance dataset diversity and model robustness:

*   **Color Jittering**: Mô phỏng điều kiện ánh sáng đa dạng trong nhà trồng nấm (thay đổi độ sáng, độ tương phản, độ bão hòa, sắc độ). Điều này giúp model học cách nhận diện bệnh dưới các điều kiện chiếu sáng khác nhau.
*   **Synthetic Data (GAN/Diffusion)**: Nếu khả thi, sẽ nghiên cứu tạo dữ liệu tổng hợp cho các trường hợp bệnh hiếm hoặc khó thu thập. Việc này sẽ được ghi chú rõ ràng trong báo cáo và có ablation study riêng để chứng minh không gây bias.
*   **Mosaic, CutMix, MixUp**: Các kỹ thuật trộn ảnh này giúp tăng cường khả năng nhận diện đối tượng trong các ngữ cảnh phức tạp, cải thiện khả năng học các đặc trưng mạnh mẽ hơn và giảm overfitting.
*   **Random Erasing/Cutout**: Giúp model học cách nhận diện đối tượng ngay cả khi một phần của chúng bị che khuất, mô phỏng các tình huống vật cản trong môi trường thực tế.
*   **Gaussian Noise/Blur**: Thêm nhiễu hoặc làm mờ nhẹ để mô phỏng điều kiện ảnh kém chất lượng hoặc rung lắc, giúp model trở nên mạnh mẽ hơn với dữ liệu thực tế.

## 3. Label Standardization and Stratified Data Splitting

1.  **Label Standardization**: Tất cả các nhãn sẽ được chuyển đổi về định dạng YOLO (`.txt`) thống nhất. Các nhãn hiện có trong dataset gốc (`Affected`, `Healthy-Affected`) sẽ được xem xét và phân loại lại thành 4 class mục tiêu: `Healthy`, `Trichoderma_spp`, `Aspergillus_spp`, `Soft_Rot`. Quá trình này có thể yêu cầu sự can thiệp thủ công hoặc bán tự động với sự kiểm tra của chuyên gia.
2.  **Stratified Split**: Dataset sẽ được chia thành tập huấn luyện (Train), kiểm định (Validation), và kiểm tra (Test) theo tỷ lệ **70%/15%/15%**. Việc chia tập dữ liệu sẽ được thực hiện theo phương pháp **stratified sampling** dựa trên phân bố của từng class. Điều này đảm bảo rằng mỗi tập con (train, val, test) có tỷ lệ các class bệnh tương tự như trong toàn bộ dataset, tránh tình trạng một class nào đó bị thiếu hụt nghiêm trọng trong tập test, đặc biệt quan trọng với dataset nhỏ và mất cân bằng.

## 4. Data Verification Flag

Trong báo cáo cuối cùng và `DATASET_CARD.md`, chúng tôi sẽ gắn cờ rõ ràng các loại ảnh:

*   **Ảnh thật đã được xác minh (Verified Real Images)**: Các ảnh được thu thập từ các nguồn đáng tin cậy và đã được kiểm tra chất lượng, nhãn chính xác. Bao gồm các ảnh từ dataset gốc sau khi đã được phân loại lại và các ảnh từ các dataset công khai khác có chất lượng cao.
*   **Ảnh sưu tầm cần chuyên gia xác nhận (Collected Images Awaiting Expert Verification)**: Các ảnh được thu thập từ các nguồn công khai nhưng chưa được chuyên gia nông nghiệp xác nhận về tính chính xác của nhãn bệnh. Các ảnh này sẽ được sử dụng trong giai đoạn huấn luyện ban đầu nhưng sẽ được đánh dấu để ưu tiên kiểm tra và xác minh bởi chuyên gia trước khi huấn luyện mô hình cuối cùng hoặc công bố.
*   **Ảnh tổng hợp (Synthetic Images)**: Nếu sử dụng dữ liệu tổng hợp từ GAN/diffusion, các ảnh này sẽ được gắn cờ rõ ràng là ảnh tổng hợp và sẽ có ablation study riêng để đánh giá tác động của chúng đến hiệu suất mô hình.
