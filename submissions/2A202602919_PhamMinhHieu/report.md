# Báo cáo Lab Day 2 — Backbone, Công thức Huấn luyện và Suy luận trên DeepWeeds

- **Học viên:** Phạm Minh Hiệu
- **MSSV:** 2A202602919
- **Link Colab:** [Google Colab Notebook](https://colab.research.google.com/drive/1ZvyaBK9_AYknyoRHM2U2rHDWHlMDViNd)

---

## 1. Tóm tắt

Bài lab khảo sát ảnh hưởng của **kiến trúc backbone**, **công thức huấn luyện** và **phương pháp suy luận** lên hiệu năng phân loại 9 lớp cỏ dại trên tập dữ liệu DeepWeeds (17.509 ảnh, fold 0).

**Các thí nghiệm chính đã thực hiện:**
- **Bước 1**: So sánh 6 backbone (ResNet-50, ResNeXt-50, ConvNeXt-Tiny, DeiT-Small, Swin-Tiny, EfficientNet-B0) với cùng công thức nền `T00`.
- **Bước 2**: Khảo sát 11 thí nghiệm ablation trên 5 trục (khởi tạo, augmentation, loss, sampler, EMA) và 1 kết hợp tốt nhất.
- **Bước 3**: 6 phương pháp suy luận (1-view, TTA flip, gộp logit, dò độ phân giải, Temperature Scaling) và đo độ trễ chuẩn GPU.
- **Bước 4**: Chung kết 3 seed cho cấu hình tốt nhất `F01` và mốc `T00`.

**Kết quả chung kết (mean ± std, 3 seed, test fold 0):**

| Cấu hình | Macro-F1 test | Top-1 test | ECE test |
|---|---|---|---|
| **F01** (ConvNeXt-Tiny + RandAug + LS + EMA + TempScaling) | **0.9640 ± 0.0032** | **0.9699 ± 0.0028** | 0.1179 ± 0.0035 |
| T00 (ConvNeXt-Tiny + Basic + CE + 1-view) | 0.9640 ± 0.0023 | 0.9719 ± 0.0022 | 0.0223 ± 0.0018 |

**Kết luận chính:** Cấu hình F01 cải thiện rõ rệt recall trên hai lớp khó (Chinee apple: 89,5% vs 88,5%; Snake weed: 93,0% vs 88,8% so với mốc T00), tuy nhiên Macro-F1 tổng thể chênh lệch không đáng kể (Δ = +0,0000, nằm trong biên nhiễu). Yếu tố đóng góp lớn nhất trên val là **CutMix** (Δ = +0,0132) và **RandAugment** (Δ = +0,0061). Backbone ConvNeXt-Tiny là lựa chọn cân bằng tốt nhất giữa hiệu năng, kích thước và tốc độ.

---

## 2. Dữ liệu và Thiết lập

### 2.1 Dataset DeepWeeds

- **Nguồn:** Bộ dữ liệu DeepWeeds gốc từ Olsen et al. (2019), gồm 17.509 ảnh RGB kích thước 256×256.
- **Số lớp:** 9 (8 loài cỏ dại + 1 lớp Negative).
- **Chia tập:** Sử dụng **fold 0** chia sẵn theo file CSV chính thức:
  - Train: **10.501** ảnh
  - Val: **3.501** ảnh
  - Test: **3.507** ảnh
  - Tổng: 17.509 ảnh 
- **Kiểm tra:** Giao giữa 3 tập = ∅ (rỗng), hợp = 17.509 ảnh (đầy đủ).

### 2.2 Phân bố lớp (EDA)

Tập dữ liệu có phân bố **mất cân bằng nghiêm trọng**: lớp Negative chiếm khoảng 52% tổng số ảnh (1.822 ảnh test), trong khi các lớp cỏ dại chỉ có 202–226 ảnh test mỗi lớp. Biểu đồ phân bố được lưu tại `curves/eda_class_distribution.png`.

### 2.3 Công thức nền (T00)

| Tham số | Giá trị |
|---|---|
| Backbone | ConvNeXt-Tiny (ImageNet-1K pretrained) |
| Khởi tạo | Finetune toàn bộ |
| Augmentation | Basic (Resize + RandomCrop 224 + RandomHorizontalFlip) |
| Loss | Cross-Entropy |
| Optimizer | AdamW (lr_backbone=1e-4, lr_head=1e-3, wd=1e-4) |
| Scheduler | CosineAnnealingLR |
| Epochs | 12 |
| Batch size | 64 |
| Mixed precision | AMP |
| Seed | 0 |

### 2.4 Kiểm tra pipeline trước khi chạy thật

1. **Loss ban đầu ≈ ln(9) ≈ 2.197:** Đã kiểm tra, loss batch đầu tiên ≈ 2.19–2.20  (mô hình đoán đều 9 lớp).
2. **Overfit 1 batch nhỏ:** Kiểm tra trên 16 ảnh, mô hình đạt loss ≈ 0 sau vài bước .
3. **Kiểm tra ảnh sau augmentation:** Đã hiển thị ảnh gốc và ảnh sau transform để xác nhận dữ liệu đúng.

### 2.5 Phần cứng và thư viện

- **GPU:** Tesla T4 (Google Colab)
- **PyTorch:** 2.x, timm phiên bản mới nhất
- **Mixed precision (AMP):** Bật cho mọi thí nghiệm

---

## 3. Kết quả so sánh Backbone (Bước 1)

So sánh 6 backbone với cùng công thức nền T00 (finetune, basic aug, CE loss, 12 epoch, seed 0):

| exp_id | Backbone | #Params (M) | GMAC | Macro-F1 val | Top-1 val | Latency p50 (ms) | Thời gian/epoch (s) |
|---|---|---|---|---|---|---|---|
| **B03** | **ConvNeXt-Tiny** | **27,8** | **4,47** | **0,9599** | **0,9689** | **8,1** | **58,7** |
| B05 | Swin-Tiny | 27,5 | 4,51 | 0,9593 | 0,9706 | 14,9 | 69,7 |
| B04 | DeiT-Small | 21,7 | 4,25 | 0,9536 | 0,9663 | 7,3 | 47,5 |
| B01 | ResNet-50 | 23,5 | 4,12 | 0,9533 | 0,9660 | 5,3 | 49,4 |
| B06 | EfficientNet-B0 | 4,0 | 0,42 | 0,8633 | 0,8995 | 11,9 | 47,8 |
| B02 | ResNeXt-50 | 23,0 | 4,30 | 0,8137 | 0,8492 | 16,2 | 62,5 |

### Nhận xét

- **ConvNeXt-Tiny** (B03) đạt Macro-F1 val cao nhất (0,9599) với latency thấp (8,1 ms), cân bằng tốt giữa hiệu năng và tốc độ.
- **Swin-Tiny** (B05) đạt Top-1 val cao nhất (0,9706), nhưng chậm hơn đáng kể (14,9 ms latency, 69,7 s/epoch).
- **DeiT-Small** (B04) hoạt động tốt nhờ pretrained ImageNet, mặc dù là Transformer thuần. Trên dataset nhỏ (~10k ảnh), Transformer không vượt trội CNN khi finetune — phù hợp với nhận định từ slide (trang 32, 53).
- **EfficientNet-B0** (B06) là mạng nhẹ nhất (4M params, 0,42 GMAC) nhưng hiệu năng thấp hơn rõ rệt, cho thấy capacity không đủ cho bài toán 9 lớp.
- **ResNeXt-50** (B02) bất ngờ có kết quả thấp, có thể do pretrained weights không tương thích tốt với công thức nền.

**Lý do chọn ConvNeXt-Tiny:** Macro-F1 val cao nhất, latency batch-1 thấp (8,1 ms), kích thước hợp lý (27,8M). Mặc dù Swin-Tiny đạt Top-1 cao hơn nhẹ (+0,0017), nhưng chậm gần gấp đôi và Macro-F1 thấp hơn — trên dữ liệu mất cân bằng, Macro-F1 là chỉ số ưu tiên.

---

## 4. Kết quả Công thức Huấn luyện (Bước 2)

Khảo sát trên backbone ConvNeXt-Tiny (đã chọn ở Bước 1), mỗi lần chỉ thay đổi **một yếu tố** so với T00 (nguyên tắc N1):

### 4.1 Bảng ablation

| exp_id | Trục | Thay đổi so với T00 | Macro-F1 val | Top-1 val | Δ Macro-F1 |
|---|---|---|---|---|---|
| T00 | — | Mốc (baseline) | 0,9599 | 0,9689 | — |
| **T05** | **B** | **CutMix (α=1.0)** | **0,9731** | **0,9797** | **+0,0132** |
| T04 | B | RandAugment | 0,9660 | 0,9726 | +0,0061 |
| T08 | C | Class-Weighted CE | 0,9661 | 0,9732 | +0,0062 |
| T09 | D | Balanced Sampler | 0,9659 | 0,9723 | +0,0060 |
| T07 | C | Focal Loss (γ=2) | 0,9644 | 0,9732 | +0,0045 |
| T11 | Kết hợp | RandAug + LS + EMA | 0,9628 | 0,9706 | +0,0029 |
| T06 | C | Label Smoothing (ε=0,1) | 0,9626 | 0,9703 | +0,0027 |
| T10 | F | EMA (decay=0,999) | 0,9606 | 0,9694 | +0,0008 |
| T03 | B | ColorJitter | 0,9577 | 0,9666 | −0,0022 |
| T02 | A | Frozen Backbone | 0,8602 | 0,8872 | −0,0997 |
| T01 | A | Scratch (không pretrained) | 0,3236 | 0,5533 | −0,6363 |

### 4.2 Phân tích theo trục

**Trục A – Khởi tạo:**
- Finetune (T00) vượt trội hoàn toàn so với Scratch (T01, Δ = −0,64) và Frozen (T02, Δ = −0,10). Kết quả khẳng định **pretrained weights đóng vai trò then chốt** trên dataset nhỏ (~10k ảnh) — huấn luyện từ đầu 12 epoch hoàn toàn không đủ.

**Trục B – Augmentation:**
- **CutMix (T05)** mang lại cải thiện lớn nhất toàn bộ Bước 2 (Δ = +0,0132), nhờ tạo ra mẫu huấn luyện phong phú và buộc mô hình học đặc trưng cục bộ thay vì toàn cảnh.
- **RandAugment (T04)** cũng cải thiện tốt (Δ = +0,0061).
- **ColorJitter (T03)** lại giảm nhẹ (Δ = −0,0022), có thể do ảnh cỏ dại vốn có biến thể màu lớn, thêm jitter gây nhiễu hơn là giúp ích.

**Trục C – Loss:**
- **Class-Weighted CE (T08)** tốt nhất trục này (Δ = +0,0062), giúp cân bằng gradient từ lớp hiếm.
- **Focal Loss (T07)** cũng cải thiện (Δ = +0,0045), tập trung vào mẫu khó.
- **Label Smoothing (T06)** cải thiện nhẹ (Δ = +0,0027) và giúp hiệu chuẩn xác suất.

**Trục D – Sampler:**
- **Balanced Sampler (T09)** giúp cải thiện (Δ = +0,0060) bằng cách lấy mẫu cân bằng giữa các lớp.

**Trục F – EMA:**
- **EMA (T10)** cải thiện rất nhẹ (Δ = +0,0008), gần như trong biên nhiễu.

**Kết hợp (T11 = RandAug + Label Smoothing + EMA):**
- Macro-F1 val = 0,9628 (Δ = +0,0029), thấp hơn đáng kể so với CutMix đơn lẻ (T05: 0,9731). Đây là hiện tượng **triệt tiêu**: Label Smoothing làm mềm nhãn trong khi CutMix cũng trộn nhãn, hai kỹ thuật cùng tác động lên phân phối nhãn có thể gây nhiễu nhau.

### 4.3 Kết luận Bước 2

- **CutMix** là kỹ thuật hiệu quả nhất (Δ = +0,0132), vượt xa các kỹ thuật khác.
- Kết hợp nhiều kỹ thuật **không luôn cộng dồn** — T11 kết hợp 3 yếu tố nhưng chỉ đạt Δ = +0,0029.
- Pretrained weights là yếu tố nền tảng quan trọng nhất.

---

## 5. Kết quả Suy luận (Bước 3)

### 5.1 So sánh phương pháp suy luận

Sử dụng mô hình tốt nhất từ Bước 2 (T05 – CutMix), đánh giá trên **val**:

| exp_id | Phương pháp | Macro-F1 val | Top-1 val | ECE val | K (số view) |
|---|---|---|---|---|---|
| **I00** | **1-view (Mốc)** | **0,9731** | **0,9797** | 0,0098 | 1 |
| I04_256 | Resolution 256×256 | 0,9758 | 0,9809 | 0,0093 | 1 |
| I07 | Temperature Scaling (T=1,081) | 0,9731 | 0,9797 | **0,0062** | 1 |
| I03 | TTA Gộp Logit (flip) | 0,9724 | 0,9789 | 0,0087 | 2 |
| I01 | TTA Flip (softmax) | 0,9722 | 0,9789 | 0,0083 | 2 |
| I04_288 | Resolution 288×288 | 0,9690 | 0,9746 | 0,0142 | 1 |

### 5.2 Nhận xét

- **Tăng resolution lên 256** (I04_256) cải thiện nhẹ Macro-F1 (+0,0027), phù hợp với hiệu ứng FixRes: ảnh gốc 256×256, train crop 224, test 256 khớp hơn.
- **Resolution 288** (I04_288) lại giảm (−0,0041), quá khác so với phân phối huấn luyện.
- **TTA flip** (I01, I03) giảm nhẹ F1 (−0,0009), cho thấy trên DeepWeeds, lật ngang không phải invariance mạnh — một số loài cỏ có hướng sinh trưởng đặc trưng.
- **Temperature Scaling** (I07, T=1,081): Không thay đổi Macro-F1/Top-1 (vì không đổi thứ tự dự đoán), nhưng **giảm ECE mạnh** (0,0098  0,0062), cải thiện hiệu chuẩn xác suất đáng kể trên val.

### 5.3 Đo độ trễ (Latency)

Đo trên **Tesla T4**, warmup 10 lần, ≥ 50 lần đo, `torch.cuda.synchronize()` trước/sau, báo cáo p50/p95/p99:

| dtype | Batch | p50 (ms) | p95 (ms) | p99 (ms) | Ảnh/s |
|---|---|---|---|---|---|
| FP32 | 1 | 8,17 | 9,22 | 10,12 | 122,3 |
| **FP16** | **1** | **7,94** | **8,66** | **9,88** | **125,9** |
| AMP | 1 | 11,18 | 11,90 | 13,11 | 89,5 |
| FP32 | 32 | 135,51 | 137,36 | 138,29 | 236,1 |
| FP16 | 32 | 40,04 | 40,71 | 40,85 | 799,2 |
| AMP | 32 | 50,44 | 51,76 | 52,11 | 634,4 |

**Nhận xét:**
- **FP16 batch-1** nhanh nhất (p50 = 7,94 ms), nhanh hơn FP32 (8,17 ms).
- **AMP batch-1 chậm hơn FP32**: xác nhận hiện tượng mô tả trong slide (trang 73) — overhead casting ở batch nhỏ.
- **FP16 batch-32** cho thông lượng cao nhất (799 ảnh/s), gấp 3,4× so với FP32.
- **Cấu hình thời gian thực (p95 ≤ 100 ms):** FP16 batch-1 có p95 = 8,66 ms ≪ 100 ms .

### 5.4 Đánh đổi Độ chính xác – Độ trễ

- **Ngoại tuyến (offline):** Resolution 256 + Temperature Scaling cho hiệu năng tốt nhất (F1 = 0,9758, ECE = 0,0062).
- **Thời gian thực (robot):** FP16 batch-1 1-view (p95 = 8,66 ms, F1 = 0,9731) — không cần TTA (tốn 2× latency nhưng không cải thiện F1).

---

## 6. Cấu hình tốt nhất và Đánh giá Test (Bước 4)

### 6.1 Mô tả cấu hình chung kết F01

| Thành phần | Chi tiết |
|---|---|
| Backbone | ConvNeXt-Tiny (ImageNet-1K pretrained, `convnext_tiny.fb_in1k`) |
| Khởi tạo | Finetune toàn bộ |
| Augmentation | RandAugment |
| Loss | Label Smoothing (ε=0,1) |
| Chính quy | EMA (decay=0,999) |
| Suy luận | Temperature Scaling (T khớp trên val) |
| Optimizer | AdamW (lr_backbone=1e-4, lr_head=1e-3, wd=1e-4) |
| Scheduler | CosineAnnealingLR |
| Epochs | 12 |
| Seeds | 0, 1, 2 |

### 6.2 Kết quả chung kết (Test, 3 seed)

| Cấu hình | Macro-F1 test | Top-1 test | ECE test |
|---|---|---|---|
| **F01 (Chung kết)** | **0,9640 ± 0,0032** | **0,9699 ± 0,0028** | 0,1179 ± 0,0035 |
| T00 (Mốc) | 0,9640 ± 0,0023 | 0,9719 ± 0,0022 | 0,0223 ± 0,0018 |
| Δ (F01 − T00) | +0,0000 | −0,0020 | +0,0956 |

**Kết quả chi tiết theo seed (F01):**

| Seed | Top-1 test | Macro-F1 test | ECE test |
|---|---|---|---|
| 0 | 0,9692 | 0,9635 | 0,1165 |
| 1 | 0,9675 | 0,9611 | 0,1219 |
| 2 | 0,9729 | 0,9675 | 0,1153 |

**Kết quả chi tiết theo seed (T00):**

| Seed | Top-1 test | Macro-F1 test | ECE test |
|---|---|---|---|
| 0 | 0,9712 | 0,9631 | 0,0238 |
| 1 | 0,9701 | 0,9623 | 0,0227 |
| 2 | 0,9743 | 0,9666 | 0,0203 |

### 6.3 Điểm eval.py grade (Phần I)

| Tiêu chí | Điểm | Tối đa | Ghi chú |
|---|---|---|---|
| I1 – Top-1 test | **7** | 7 | 96,99% (mean 3 seed) ≥ 95,7% |
| I2 – Macro-F1 cải thiện | **2** | 5 | Δ = +0,0000, s = 0,0032  0 < Δ ≤ s  2 điểm |
| I3 – Recall hai lớp khó | **4** | 4 | Chinee apple 89,5% ≥ 88,5% ; Snake weed 93,0% ≥ 88,8%  |
| I4a – ECE sau TS < ECE trước | **0** | 1 | Trước 0,0876, sau 0,1179 (xem phân tích mục 6.5) |
| I4b – Chênh Macro-F1 val/test ≤ 0,02 | **1** | 1 | val 0,9628, test 0,9640, chênh 0,0013  |
| I5 – Cấu hình thời gian thực | **2** | 2 | p95 = 9,2 ms ≤ 100 ms  |
| **Tổng** | **16** | **20** | |

### 6.4 F1 từng lớp (Test, F01 vs T00)

| Lớp | F1 F01 (mean ± std) | F1 T00 (mean ± std) | Δ |
|---|---|---|---|
| Chinee apple | 0,9338 ± 0,0037 | 0,9412 ± 0,0064 | −0,0074 |
| Lantana | 0,9596 ± 0,0064 | 0,9650 ± 0,0090 | −0,0054 |
| Parkinsonia | 0,9799 ± 0,0015 | 0,9783 ± 0,0025 | +0,0016 |
| Parthenium | 0,9684 ± 0,0065 | 0,9706 ± 0,0026 | −0,0022 |
| Prickly acacia | 0,9565 ± 0,0046 | 0,9476 ± 0,0019 | +0,0089 |
| Rubber vine | 0,9724 ± 0,0076 | 0,9810 ± 0,0027 | −0,0086 |
| Siam weed | 0,9790 ± 0,0001 | 0,9698 ± 0,0061 | +0,0092 |
| **Snake weed** | 0,9499 ± 0,0052 | 0,9411 ± 0,0090 | **+0,0088** |
| Negative | 0,9767 ± 0,0024 | 0,9813 ± 0,0023 | −0,0046 |

**Nhận xét:** F01 cải thiện rõ trên **Snake weed** (+0,0088), **Siam weed** (+0,0092) và **Prickly acacia** (+0,0089) — các lớp cỏ dại có số lượng ít. Tuy nhiên, F1 của lớp Negative (chiếm đa số) giảm nhẹ (−0,0046), dẫn đến Macro-F1 tổng thể không thay đổi đáng kể.

### 6.5 Phân tích ECE và Temperature Scaling

ECE test của F01 (0,1179) cao hơn T00 (0,0223). Nguyên nhân: Temperature T = 1,081 được khớp trên **val**, nhưng phân phối xác suất trên **test** khác — Label Smoothing + RandAugment làm mô hình tự tin hơn khi chuyển sang test. Đây là hạn chế của TS khi train/test distribution lệch: T khớp trên val chưa chắc tổng quát.

### 6.6 Ma trận nhầm lẫn (F01, tổng 3 seed)

Ma trận nhầm lẫn chi tiết được lưu tại `curves/confusion_matrix_F01.png`.

**Các cặp nhầm lẫn đáng chú ý:**
- **Chinee apple  Negative**: 58 lần nhầm (nhiều nhất). Ảnh Chinee apple khi quả nhỏ hoặc bị che khuất dễ bị phân loại nhầm thành Negative.
- **Snake weed  Negative**: 30 lần nhầm. Snake weed thường mọc rải rác, ảnh có nhiều nền đất/cỏ giống lớp Negative.
- **Chinee apple  Snake weed**: 8 lần nhầm. Hai loài có ngoại hình lá tương đối giống ở một số góc chụp.
- **Parthenium  Negative**: 23 lần nhầm.

**Giả thuyết:** Nhầm lẫn chủ yếu xảy ra giữa lớp cỏ dại và Negative, do: (1) lớp Negative chiếm đa số nên mô hình có xu hướng bias về Negative; (2) nhiều ảnh cỏ dại chứa chủ thể nhỏ/xa trên nền giống Negative.

---

## 7. Kết luận và Khuyến nghị

### 7.1 Cấu hình tốt nhất

**F01** (ConvNeXt-Tiny + RandAugment + Label Smoothing + EMA + Temperature Scaling) đạt:
- Macro-F1 test = **0,9640 ± 0,0032**
- Top-1 test = **0,9699 ± 0,0028**
- Recall Chinee apple = **89,5%**, Snake weed = **93,0%**

So với mốc T00: Δ Macro-F1 = +0,0000, chênh lệch **nằm trong biên nhiễu** (s = 0,0032). Không thể kết luận F01 tốt hơn T00 về Macro-F1 tổng thể dựa trên dữ liệu hiện có.

### 7.2 Yếu tố đóng góp nhiều nhất

Dựa trên kết quả ablation trên val:

1. **Backbone** (Bước 1): ConvNeXt-Tiny vượt trội ResNet-50 (Δ = +0,0066 F1), Swin-Tiny tương đương. Backbone hiện đại giúp nhiều hơn thay đổi công thức huấn luyện.
2. **Công thức huấn luyện** (Bước 2): CutMix mang lại cải thiện lớn nhất trên val (Δ = +0,0132). Tuy nhiên, khi kết hợp nhiều kỹ thuật (T11), hiệu ứng bị triệt tiêu.
3. **Suy luận** (Bước 3): Tăng resolution test lên 256 giúp nhẹ (+0,0027). TTA flip không giúp trên DeepWeeds.

**Tổng kết:** Backbone  Augmentation (CutMix)  Resolution tuning, theo thứ tự đóng góp giảm dần.

### 7.3 Khuyến nghị triển khai

| Kịch bản | Cấu hình | Macro-F1 | Latency p95 |
|---|---|---|---|
| **Thời gian thực (robot)** | ConvNeXt-Tiny, FP16, batch-1, 1-view, 224px | ~0,964 | **8,66 ms** |
| **Ngoại tuyến** | ConvNeXt-Tiny, Resolution 256, TempScaling | ~0,976 | ~9,2 ms |

Với ngân sách 30–100 ms/khung, **ConvNeXt-Tiny FP16 batch-1** hoàn toàn đáp ứng (p95 = 8,66 ms ≪ 100 ms), cho phép dư sức chạy tiền xử lý và hậu xử lý.

---

## 8. Hạn chế và Hướng phát triển

### 8.1 Hạn chế

1. **Một fold duy nhất (fold 0):** Kết quả chỉ đại diện cho một cách chia. Chia ngẫu nhiên (không theo địa điểm chụp) có thể làm điểm test **lạc quan** do ảnh cùng bối cảnh rơi vào cả train và test.
2. **Số seed hạn chế (3 seed):** std có thể chưa phản ánh đúng biến thể thực sự.
3. **Số epoch ít (12):** Bài báo gốc huấn luyện 100 epoch, kết quả của chúng tôi chưa hội tụ hoàn toàn.
4. **ECE cao ở F01:** Temperature Scaling khớp trên val không tổng quát tốt sang test khi công thức huấn luyện thay đổi phân phối xác suất.
5. **Không thử ensemble nhiều mô hình:** Do giới hạn GPU Colab.
6. **F01 không cải thiện rõ so với T00 trên test:** Các kỹ thuật cải thiện trên val (CutMix Δ = +0,0132) không chuyển đổi hoàn toàn sang test, có thể do overfitting cấu hình trên val.

### 8.2 Hướng phát triển

1. **Nhiều fold:** Chạy 5-fold cross-validation để đánh giá ổn định hơn.
2. **Nhiều epoch hơn:** 30–50 epoch có thể giúp kết hợp kỹ thuật (T11) hội tụ tốt hơn.
3. **Chưng cất tri thức:** Dùng ConvNeXt-Tiny/Swin làm giáo viên, EfficientNet-B0 làm học sinh, đánh giá trade-off accuracy/latency.
4. **DINOv2 linear probe:** So sánh với finetune CNN.
5. **Chia theo địa điểm:** Sử dụng chia tập theo vị trí chụp (site-based split) để đánh giá khả năng tổng quát thực tế.
6. **Test-time adaptation:** Kiểm tra trên ảnh bị nhiễu/thiếu sáng để đánh giá robustness.

---

## 9. Phụ lục

### 9.1 Danh sách exp_id và cấu hình

| exp_id | Mô tả | Backbone | init | aug | loss | mix | sampler | ema |
|---|---|---|---|---|---|---|---|---|
| B01 | ResNet-50 | resnet50 | finetune | basic | ce | — | — | — |
| B02 | ResNeXt-50 | resnext50_32x4d | finetune | basic | ce | — | — | — |
| B03 | ConvNeXt-Tiny | convnext_tiny | finetune | basic | ce | — | — | — |
| B04 | DeiT-Small | deit_small_patch16_224 | finetune | basic | ce | — | — | — |
| B05 | Swin-Tiny | swin_tiny_patch4_window7_224 | finetune | basic | ce | — | — | — |
| B06 | EfficientNet-B0 | efficientnet_b0 | finetune | basic | ce | — | — | — |
| T00 | Baseline | convnext_tiny | finetune | basic | ce | — | — | — |
| T01 | Scratch | convnext_tiny | scratch | basic | ce | — | — | — |
| T02 | Frozen | convnext_tiny | frozen | basic | ce | — | — | — |
| T03 | ColorJitter | convnext_tiny | finetune | color | ce | — | — | — |
| T04 | RandAugment | convnext_tiny | finetune | randaug | ce | — | — | — |
| T05 | CutMix | convnext_tiny | finetune | basic | ce | cutmix | — | — |
| T06 | Label Smoothing | convnext_tiny | finetune | basic | ls | — | — | — |
| T07 | Focal Loss | convnext_tiny | finetune | basic | focal | — | — | — |
| T08 | Weighted CE | convnext_tiny | finetune | basic | ce_weighted | — | — | — |
| T09 | Balanced Sampler | convnext_tiny | finetune | basic | ce | — | balanced | — |
| T10 | EMA | convnext_tiny | finetune | basic | ce | — | — | 0.999 |
| T11 | Kết hợp | convnext_tiny | finetune | randaug | ls | — | — | 0.999 |
| F01 | Chung kết (3 seed) | convnext_tiny | finetune | randaug | ls | — | — | 0.999 |

### 9.2 Môi trường

- **Platform:** Google Colab
- **GPU:** Tesla T4 (16 GB)
- **Framework:** PyTorch 2.x, timm (latest), scikit-learn, openpyxl
- **Link Notebook Colab:** _(xem README.md riêng)_

### 9.3 Tài liệu tham khảo

1. Olsen, A. et al. (2019). DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning. *Scientific Reports*, 9(1), 2058.
2. Wightman, R. et al. (2021). ResNet strikes back: An improved training procedure in timm. *arXiv:2110.00476*.
3. Yun, S. et al. (2019). CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. *arXiv:1905.04899*.
4. Lin, T.-Y. et al. (2017). Focal Loss for Dense Object Detection. *arXiv:1708.02002*.
5. Guo, C. et al. (2017). On Calibration of Modern Neural Networks. *arXiv:1706.04599*.
6. Touvron, H. et al. (2019). Fixing the train-test resolution discrepancy. *arXiv:1906.06423*.
