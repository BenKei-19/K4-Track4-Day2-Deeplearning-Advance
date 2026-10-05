# Lab Day 2 — Submission README

## Thông tin

- **Học viên:** Phạm Minh Hiệu
- **MSSV:** 2A202602919
- **Bài lab:** Backbone, Công thức Huấn luyện và Suy luận trên DeepWeeds
- **Dataset:** DeepWeeds (17.509 ảnh, 9 lớp, fold 0)
- **Nền tảng:** Google Colab (T4 GPU)

## Link Notebook chạy lại

> **Google Colab Notebook:** [Mở trên Colab](https://colab.research.google.com/drive/1ZvyaBK9_AYknyoRHM2U2rHDWHlMDViNd)
>
> Hoặc tải file `code/lab_day2_colab.ipynb` lên Colab, chọn Runtime → Change runtime type → T4 GPU, rồi chạy tuần tự từ trên xuống dưới.

## Phiên bản thư viện

| Thư viện | Phiên bản |
|---|---|
| Python | 3.13.x |
| PyTorch | 2.x (CUDA) |
| timm | latest |
| scikit-learn | latest |
| openpyxl | latest |
| matplotlib | latest |
| gdown | latest |

## Cách chạy lại

1. Mở file `code/lab_day2_colab.ipynb` trên Google Colab.
2. Chọn **Runtime  Change runtime type  T4 GPU**.
3. Chạy tuần tự **tất cả các cell** từ trên xuống dưới. Notebook tự động:
   - Cài đặt thư viện cần thiết.
   - Tải dataset DeepWeeds từ Google Drive (hoặc fallback Zenodo).
   - Giải nén và tự động phát hiện đường dẫn ảnh.
   - Tải labels CSV từ GitHub chính thức.
   - Chạy toàn bộ pipeline 5 bước.
4. Thời gian chạy ước tính: **~3-4 giờ** trên T4 GPU.

## Seeds đã dùng

- **Bước 1 & 2:** `seed=0` cho tất cả thí nghiệm (B01–B06, T00–T11).
- **Bước 4 (Chung kết):** `seed=0, 1, 2` cho F01 và T00.

## Cấu trúc thư mục nộp bài

```
submission/
├── README.md               # File này
├── report.md               # Báo cáo kết luận (4-8 trang)
├── results.xlsx            # Bảng số liệu 7 sheets
├── curves/                 # 21 biểu đồ training + EDA + confusion matrix
│   ├── eda_class_distribution.png
│   ├── B01_resnet50.png
│   ├── B02_resnext50_32x4d.png
│   ├── B03_convnext_tiny.png
│   ├── B04_deit_small_patch16_224.png
│   ├── B05_swin_tiny_patch4_window7_224.png
│   ├── B06_efficientnet_b0.png
│   ├── T00_convnext_tiny.png ~ T11_convnext_tiny.png
│   ├── F01_convnext_tiny.png
│   └── confusion_matrix_F01.png
├── predictions/            # 32 file dự đoán (val + test, các seed)
│   ├── F01_seed{0,1,2}_test.csv      # Chung kết test
│   ├── F01_seed{0,1,2}_val.csv       # Chung kết val
│   ├── F01uncal_seed{0,1,2}_test.csv # Chưa hiệu chuẩn
│   ├── T00_seed{0,1,2}_test.csv      # Mốc test
│   ├── T00_seed{0,1,2}_val.csv       # Mốc val
│   ├── B01~B06_seed0_val.csv         # Backbone val
│   └── T01~T11_seed0_val.csv         # Training val
├── eval_out/               # Kết quả eval.py score & grade
│   ├── F01_summary.json
│   ├── T00_summary.json
│   ├── F01_per_class.csv
│   ├── F01_per_seed.csv
│   ├── F01_confusion_sum.csv
│   └── grade_I.json
└── code/
    └── lab_day2_colab.ipynb  # Notebook hoàn chỉnh
```

## Kết quả chính

| Cấu hình | Macro-F1 test (mean ± std) | Top-1 test (mean ± std) |
|---|---|---|
| **F01** (ConvNeXt-Tiny + RandAug + LS + EMA + TS) | **0.9640 ± 0.0032** | **0.9699 ± 0.0028** |
| T00 (Mốc) | 0.9640 ± 0.0023 | 0.9719 ± 0.0022 |

**Điểm eval.py grade (Phần I): 16/20**

## Ghi chú

- Không sửa file `eval.py`.
- Không commit dataset hoặc checkpoint lớn.
- Mọi số liệu trong `results.xlsx` và `report.md` đến từ lần chạy thật, truy ngược được qua `eval_out/` và `predictions/`.
