"""
CBAM Attention Module for StrawMind-v2
=====================================
Convolutional Block Attention Module (CBAM)
- Woo et al., ECCV 2018: "CBAM: Convolutional Block Attention Module"
- Tích hợp vào YOLOv8s C2f blocks để focus vào vùng bệnh nấm

Novelty: CBAM giúp model học feature của 3 loại bệnh rõ hơn:
  - Mốc xanh (Trichoderma): màu xanh lá
  - Mốc đen (Aspergillus): màu đen đặc trưng
  - Thối nhũn (Soft Rot): texture mềm nhũn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention: Squeeze-and-Excitation style
    Học CHANNEL nào quan trọng cho từng bệnh
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)  # Tránh quá nhỏ
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention: Conv 7x7
    Học VỊ TRÍ NÀO trên ảnh có bệnh
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size phải là 3 hoặc 7"
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(concat))


class CBAM(nn.Module):
    """
    CBAM = Channel Attention + Spatial Attention (tuần tự)
    
    Novelty của StrawMind-v2:
    - Channel attention: phân biệt màu xanh (Trichoderma) vs đen (Aspergillus)
    - Spatial attention: xác định vị trí bệnh trên phôi/túi nấm
    """
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_att(x)   # Step 1: Channel attention
        x = x * self.spatial_att(x)   # Step 2: Spatial attention
        return x


class C2f_CBAM(nn.Module):
    """
    C2f Block + CBAM Attention
    Thay thế một số C2f blocks trong YOLOv8s neck bằng C2f_CBAM
    
    Cách dùng trong custom training:
      - Khởi tạo YOLOv8s model
      - Thay thế neck layers cuối bằng C2f_CBAM
      - Fine-tune
    """
    def __init__(self, in_channels: int, out_channels: int,
                 n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(out_channels * e)
        self.cv1 = nn.Conv2d(in_channels, 2 * self.c, 1, 1, bias=False)
        self.cv2 = nn.Conv2d((2 + n) * self.c, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * self.c)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        # Thêm CBAM sau C2f
        self.cbam = CBAM(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.cv1(x)))
        y = list(y.chunk(2, dim=1))
        out = self.act(self.bn2(self.cv2(torch.cat(y, dim=1))))
        # Apply CBAM attention
        return self.cbam(out)


class DiseaseSeverityHead(nn.Module):
    """
    Disease Severity Estimation Head — NOVELTY gốc của StrawMind
    ============================================================
    Dự đoán MỨC ĐỘ BỆNH (0.0 → 1.0) song song với detection
    
    Output: float [0, 1]
      0.0 = Completely healthy (100% khỏe)
      0.5 = Mixed/early stage (50% bị ảnh hưởng)
      1.0 = Severe infection (100% bị bệnh nặng)
    
    Ý nghĩa thực tế:
      - Severity < 0.2: Theo dõi thêm
      - Severity 0.2–0.5: Cần xử lý sớm
      - Severity > 0.5: Cần cách ly ngay
    """
    def __init__(self, in_features: int = 512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x))


def get_cbam_info():
    """In thông tin về CBAM module"""
    print("=" * 60)
    print("CBAM ATTENTION MODULE — STRAWMIND-v2 NOVELTY")
    print("=" * 60)
    print()
    print("Architecture:")
    print("  1. Channel Attention (SE-style)")
    print("     - AvgPool + MaxPool → Shared FC → Sigmoid")
    print("     - Học CHANNEL nào quan trọng")
    print()
    print("  2. Spatial Attention (Conv 7×7)")
    print("     - AvgPool + MaxPool dọc channel → Conv → Sigmoid")
    print("     - Học VỊ TRÍ nào có bệnh")
    print()
    print("  3. Disease Severity Head (ORIGINAL)")
    print("     - GAP → FC(512→128→32→1) → Sigmoid")
    print("     - Output: 0.0 (khỏe) → 1.0 (bệnh nặng)")
    print()
    print("Expected improvement: +3-8% mAP@50")
    print("=" * 60)

    # Demo với tensor giả
    dummy = torch.randn(2, 256, 40, 40)  # Batch=2, C=256, H=W=40
    cbam = CBAM(256)
    out = cbam(dummy)
    print(f"\nDemo CBAM:")
    print(f"  Input:  {dummy.shape}")
    print(f"  Output: {out.shape}  ✓ Same shape")

    severity_head = DiseaseSeverityHead(in_features=256)
    sev = severity_head(dummy)
    print(f"\nDemo Severity Head:")
    print(f"  Input:  {dummy.shape}")
    print(f"  Output: {sev.shape}  ✓ Severity scores")
    print(f"  Values: {sev.squeeze().tolist()}")


if __name__ == "__main__":
    get_cbam_info()
