# 🇬🇧 🇻🇳 Đồ Án Dịch Máy Anh - Việt (English to Vietnamese Machine Translation)

Dự án nghiên cứu và Fine-tune mô hình ngôn ngữ tiền huấn luyện (Pre-trained Language Models) cho tác vụ dịch máy Anh - Việt, sử dụng mô hình MarianMT (`Helsinki-NLP/opus-mt-en-vi`) trên các tập dữ liệu **PhoMT** và **IWSLT15**.

---

## 📁 Cấu Trúc Thư Mục (Directory Structure)

```text
.
├── data/               # Chứa dữ liệu thô và dữ liệu đã xử lý (IWSLT15, PhoMT)
├── notebooks/          # Chứa các file Jupyter / Colab notebook thử nghiệm & huấn luyện
│   └── baseline_opus_mt  # Notebook kiểm tra GPU, cài thư viện & test baseline
├── src/                # Mã nguồn xử lý dữ liệu, huấn luyện và đánh giá
├── reports/            # Chứa đề cương, báo cáo đồ án và tài liệu tham khảo
├── requirements.txt    # Danh sách các thư viện Python cần thiết
└── README.md           # Tài liệu hướng dẫn dự án
```

---

## 🛠️ Cài Đặt Môi Trường (Setup & Installation)

### 1. Yêu cầu hệ thống
- **Python**: `>= 3.10`
- **GPU**: Khuyên dùng GPU Tesla T4 / P100 (Google Colab hoặc Kaggle)

### 2. Cài đặt các thư viện phụ thuộc

Tạo file `requirements.txt` với nội dung:
```text
torch
transformers
datasets
sacrebleu
evaluate
sentencepiece
sacremoses
```

Cài đặt bằng câu lệnh:
```bash
pip install -r requirements.txt
```

---

## 🚀 Hướng Dẫn Chạy Thử (Quick Start)

### Chạy Notebook Baseline trên Google Colab / Kaggle

1. Tải notebook `notebooks/01_baseline_check.ipynb` lên Google Colab hoặc Kaggle.
2. Kích hoạt GPU cho môi trường thực thi:
   - **Google Colab:** `Runtime` $\rightarrow$ `Change runtime type` $\rightarrow$ Chọn `T4 GPU`.
   - **Kaggle:** `Notebook options` $\rightarrow$ `Accelerator` $\rightarrow$ Chọn `GPU T4 x2`.
3. Chạy toàn bộ các cell trong notebook để:
   - Kiểm tra thông số phần cứng GPU (`torch.cuda`).
   - Kiểm tra phiên bản và môi trường của các thư viện NLP.
   - Thử nghiệm mô hình baseline `Helsinki-NLP/opus-mt-en-vi` với danh sách câu mẫu Anh – Việt.

---

## 💻 Đoạn Mã Thử Nghiệm Baseline (Code Snippet)

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Khởi tạo mô hình và thiết bị
model_name = "Helsinki-NLP/opus-mt-en-vi"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 2. Dữ liệu thử nghiệm
english_sentences = [
    "Artificial Intelligence is transforming the world.",
    "Machine translation helps bridge the language gap between cultures.",
    "Learning Python programming is very useful for data science."
]

# 3. Tiến hành dịch thử với Beam Search
for i, src_text in enumerate(english_sentences, 1):
    inputs = tokenizer(src_text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            max_length=128
        )
    
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    print(f"[{i}] En: {src_text}\n    Vi: {translated_text}\n")
```

---

## 📊 Dữ Liệu Huấn Luyện (Datasets)

- **IWSLT15 (En-Vi):** Tập dữ liệu song ngữ từ bài nói TED Talks, dùng để xây dựng và đánh giá baseline.
- **PhoMT:** Tập dữ liệu dịch máy Anh - Việt quy mô lớn, đa ngữ cảnh cho các mô hình dịch máy chuyên sâu.

---

## 👥 Thành Viên Thực Hiện (Team Members)
* **Phan Đình Hiếu**
* **Dương Thạch Phú**
* **Trần Thiên Phú**
* **Kiều Hoài Nam**