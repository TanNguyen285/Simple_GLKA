# CONTEXT.MD — Nghiên Cứu Traffic State Classification
> Tác giả: Nguyễn Hoàng Anh Tân | Cập nhật: 2026

---

## 1. TỔNG QUAN DỰ ÁN

### Bài toán
Phân loại mật độ giao thông thời gian thực từ camera cố định — **3 class**:

| Class | Mô tả | Đặc trưng pixel-level |
|---|---|---|
| `0_clear` | Thông thoáng, xe di chuyển tự do | Pixel motion lớn, mật độ thấp |
| `1_false_jam` | Kẹt giả — đông nhưng vẫn di chuyển liên tục | Pixel motion trung bình, mật độ cao liên tục |
| `2_true_jam` | Kẹt thật — xe gần như đứng yên ≥ 3-5 giây | Pixel motion ≈ 0, mật độ cao, ít thay đổi |

### Edge case quan trọng
- **Xe lớn tạo khoảng trống tạm thời**: xe tải/bus không lấp đầy khoảng trống nhanh như xe máy → model dễ nhầm sang `clear` nếu chỉ nhìn 1 frame
- **Single frame không đủ** để phân biệt `false_jam` vs `true_jam` — bắt buộc phải có temporal information

### Triết lý thiết kế (bất biến)
- **End-to-End Deep Learning** — không dùng Object Detection (YOLO, SSD...), không dùng Object Tracking (ByteTrack, DeepSORT...)
- Mô hình tự học Spatiotemporal Features từ raw pixel
- Target hardware: **RTX 4050** (primary), **Raspberry Pi 5** (secondary, đã đạt 40.4 FPS với v1)
- **Generalization** ưu tiên — dùng được trên nhiều đoạn đường, không tied với 1 scene

---

## 2. CÔNG TRÌNH ĐÃ HOÀN THÀNH — Simple_GLKA (v1)

### 2.1 Module GLKA

```
x → DW5×5 → g
              ├── SE block → A_ch = SE(g) ⊗ g
              └── 4 dilated branches → A_sp = Σ BN_i(branch_i(g))
              y = A_ch ⊗ A_sp
```

| Nhánh | Kernel | Dilation | ERF |
|---|---|---|---|
| branch1 | 3×3 | 1 | 3×3 |
| branch2 | 3×3 | 3 | 7×7 |
| branch3 | 5×5 | 2 | 9×9 |
| branch4 | 5×5 | 3 | 13×13 |

**Reparameterization:** 4 nhánh → merge thành 1 DWConv 13×13. Sai số < 10⁻⁵.
**Lý thuyết:** HDC (Wang et al. 2018) — M² ≤ K để tránh gridding.

### 2.2 Backbone Simple_GLKA

```
stem: Conv3×3 s=2 → 32ch          [112×112×32]
B1:   EfficientBlock s=1           [112×112×32]
B2:   EfficientBlock+GLKA s=2      [56×56×64]
B3:   EfficientBlock+GLKA s=1      [56×56×64]
B4:   EfficientBlock+GLKA s=2      [28×28×128]
B5:   EfficientBlock s=1           [28×28×128]
B6:   EfficientBlock s=2           [14×14×256]
GAP → Dropout(0.3) → FC(256→num_classes)
```

**EfficientBlock:** Inverted residual (MobileNetV2 style), t=2. GLKA thay DWConv khi `use_glka=True`.

### 2.3 Kết quả (2-class: kẹt / không kẹt)

| Mô hình | Accuracy | Val_loss | Params |
|---|---|---|---|
| CNN Baseline | 98.09% | 0.0930 | 0.234M |
| GLKA+ (fusion cộng) | 98.39% | 0.0811 | 0.278M |
| **GLKA× (fusion nhân)** | **98.99%** | **0.0370** | **0.278M** |

**Inference:** RTX 4050 → 460 FPS | Pi 5 → 40.4 FPS (reparam + ONNX)

### 2.4 Giới hạn
- Single frame → không phân biệt false_jam vs true_jam
- 2 class → chưa encode velocity

---

## 3. GLKA-FLOW v2

### 3.1 Mục tiêu
- 3-class spatiotemporal: phân biệt clear / false_jam / true_jam
- Generalize qua nhiều scene/đoạn đường
- Real-time trên RTX 4050 và Pi 5

### 3.2 Kiến trúc: 4 khối

```
frame_t  → AppearanceBackbone → fa (B, 128, 28, 28)
diff_t   → VelocityBranch     → fb (B, 128, 28, 28)
                                        ↓
                              MergeFusion (concat+1×1)
                                        ↓
                              ClassificationHead
                                        ↓
                                 3-class output
```

**Lý do Dual-Branch thay vì Early Fusion (4ch stack):**
Early fusion entangle appearance + motion ngay từ stem → tied với texture scene → kém generalize khi đổi đoạn đường. Branch B chỉ thấy diff = pure motion signal → scene-invariant.

### 3.3 AppearanceBackbone

Giữ nguyên Simple_GLKA v1, cắt tại B4:

```
stem: Conv3×3 s=2, 3→32ch
B1:   EfficientBlock s=1        [112×112×32]
B2:   EfficientBlock+GLKA s=2   [56×56×64]
B3:   EfficientBlock+GLKA s=1   [56×56×64]
B4:   EfficientBlock+GLKA s=2   [28×28×128]  ← output F_A
```

### 3.4 VelocityBranch

**Input:** diff_frame `(B, 3, 224, 224)` = `frame_t - frame_{t-1}`, clipped [-1,1]

**Block:** VelocityDWBlock = PW(1×1) + DWConv(3×3) + BN + ReLU6

```
stem:   Conv3×3 s=2, 3→16ch     [112×112×16]
DWB1:   PW(16→32)  + DW s=2     [56×56×32]
DWB2:   PW(32→64)  + DW s=2     [28×28×64]
DWB3:   PW(64→128) + DW s=1     [28×28×128]  ← output F_B
```

**Lý do PW+DW:** DWConv thuần không đổi được số channel. PW tăng channel, DW làm spatial filtering — tách biệt, không có skip connection, không SE, không GLKA.

### 3.5 MergeFusion

```
concat(F_A, F_B) → (B, 256, 28, 28)
1×1 Conv + BN + ReLU6 → (B, 128, 28, 28)
```

**Lý do concat thay vì add:** add buộc 2 branch học cùng feature space — không hợp lý vì appearance ≠ motion. Concat để Head tự quyết định tỷ lệ dùng mỗi branch.

### 3.6 ClassificationHead

```
B5: EfficientBlock s=1  [28×28×128]
B6: EfficientBlock s=2  [14×14×256]
GAP → Dropout(0.3) → FC(256→3)
```

Tái dùng B5-B6 từ Simple_GLKA v1.

### 3.7 Params

| Khối | Params |
|---|---|
| AppearanceBackbone | 0.100M |
| VelocityBranch | 0.014M |
| MergeFusion | 0.033M |
| ClassificationHead | 0.172M |
| **Total** | **0.319M** |

### 3.8 Input Pipeline Runtime

```python
diff = frame_curr.astype(float) - frame_prev.astype(float)
diff = np.clip(diff / 255.0, -1, 1)
# AppearanceBackbone ← frame_curr (normalized RGB)
# VelocityBranch     ← diff
```

Sliding window = 2 frames. Mỗi frame mới → 1 kết quả ngay. Không bottleneck FPS.

### 3.9 Data Collection

- Video camera cố định, **nhiều đoạn đường khác nhau** (generalization)
- Extract frame pairs `(frame_t, frame_{t-1})` với label 3 class
- 1 video 30fps × 10s ≈ 299 sample pairs

| Class | Tiêu chí label |
|---|---|
| `0_clear` | Xe di chuyển tự do, diff lớn lan rộng |
| `1_false_jam` | Đông xe, diff trung bình đều khắp |
| `2_true_jam` | Xe đứng yên ≥ 3-5s, diff gần như toàn đen |

---

## 4. LITERATURE MAP

### 4.1 Nền tảng Two-Stream / Dual-Branch

| Bài báo | Venue | Đóng góp liên quan |
|---|---|---|
| Simonyan & Zisserman — *Two-Stream CNNs* | NIPS 2014 | Nền tảng dual-branch RGB + Flow; motion stream generalizes tốt hơn RGB đơn thuần |
| Islam et al. — *Efficient Two-Stream (SepConvLSTM + MobileNet)* | ICIP 2021, arXiv 2102.10590 | **Closest prior work**: Stream 2 = diff adjacent frames + MobileNet trên surveillance camera cố định. Họ dùng SepConvLSTM (overhead Pi) — v2 bỏ LSTM, dùng DWBlock thuần |
| Moreira et al. — *2s-MDCN* | arXiv 2211.04255, 2022 | Two-stream không LSTM: 80fps Jetson Nano. Xác nhận dual-branch không LSTM đủ nhanh cho edge |
| Lai et al. — *End-to-End Two-Stream with Representation Flow* | arXiv 2411.18002, 2024 | Replace optical flow bằng learned representation end-to-end |

### 4.2 Motion Feature Learning

| Bài báo | Venue | Đóng góp liên quan |
|---|---|---|
| Piergiovanni & Ryoo — *Representation Flow* | CVPR 2019, arXiv 1810.01455 | Lớp conv fully-differentiable học flow bên trong CNN, scene-invariant |
| Fan et al. — *TVNet* | CVPR 2018, arXiv 1804.00413 | Unfold TV-L1 thành neural layers, học optical-flow-like features end-to-end |
| Kwon et al. — *MotionSqueeze* | ECCV 2020, arXiv 2007.09933 | Correlation tensor giữa 2 frame → motion feature qua DW-separable. **Ablation candidate** thay diff frame |

### 4.3 VelocityBranch — DWConv Design

| Bài báo | Venue | Đóng góp liên quan |
|---|---|---|
| Chollet — *Xception* | CVPR 2017 | DW-separable: tách spatial filtering (DW) + channel mixing (PW) |
| Sandler et al. — *MobileNetV2* | CVPR 2018 | Inverted residual + DWConv — reference kiến trúc |
| Luo et al. — *Frame Difference + Conv* | Sensors 2025 | FDM: diff + conv extract spatiotemporal features từ adjacent frames |

### 4.4 Nền tảng GLKA (đã cite)

| Paper | Venue | Vai trò |
|---|---|---|
| RepVGG — Ding et al. | CVPR 2021 | Structural reparameterization |
| RepLKNet — Ding et al. | CVPR 2022 | Large kernel + reparameterization |
| VAN — Guo et al. | 2023 | Large Kernel Attention gốc |
| CBAM — Woo et al. | ECCV 2018 | Dual attention channel + spatial |
| SENet — Hu et al. | CVPR 2018 | Channel attention |
| HDC — Wang et al. | WACV 2018 | Hybrid dilated convolution, lý thuyết gridding |
| MobileNetV2 — Sandler et al. | CVPR 2018 | Inverted residual |
| SimpleEfficientCNN — Wang et al. | Agriculture 2026 | Backbone gốc |

---

## 5. QUYẾT ĐỊNH KỸ THUẬT ĐÃ CHỐT

| Vấn đề | Quyết định | Lý do |
|---|---|---|
| Temporal input | diff frame `frame_t - frame_{t-1}` | Zero overhead, end-to-end, camera cố định → diff = pure motion |
| Fusion strategy | Mid fusion (concat + 1×1) | Appearance ≠ motion → không thể add, concat để Head tự cân bằng |
| Early fusion (4ch) | Loại | Kém generalize khi đổi scene |
| TSM | Loại | Stack T frames tốn memory Pi 5 |
| LSTM | Loại | Overhead quá lớn trên Pi 5 |
| Velocity Block | PW + DWConv + BN + ReLU6 | DW thuần không đổi channel được |
| Fusion point | 28×28, 128ch | Đủ nhỏ cho compute, còn đủ spatial info |
| Num classes | 3 (clear / false_jam / true_jam) | Velocity phải là class riêng |
| Dataset | Nhiều đoạn đường khác nhau | Generalization |

---

## 6. VIỆC CẦN LÀM TIẾP THEO

- [x] Thiết kế và implement GLKA-Flow v2 (`glka_flow_v2.py`)
- [x] Verify: shapes, reparam, params
- [ ] Training script (dataloader frame pairs, class weighting, load v1 weights)
- [ ] Data collection script (extract frame pairs từ video, label tool)
- [ ] Ablation: diff frame vs MotionSqueeze | concat vs add vs SE-gate ở Merge
- [ ] Benchmark FPS trên RTX 4050 và Pi 5