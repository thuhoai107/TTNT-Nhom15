# =====================================================
# CHƯƠNG 2: TIỀN XỬ LÝ DỮ LIỆU
# Đề tài:
# Ứng dụng mô hình PhoBERT trong phát hiện và phân loại
# bình luận độc hại trên các nền tảng mạng xã hội ở Việt Nam
# =====================================================

import pandas as pd
import re

from pyvi import ViTokenizer
from transformers import AutoTokenizer

# =====================================================
# 2.2 KHẢO SÁT DỮ LIỆU
# =====================================================

print("=" * 60)
print("CHƯƠNG II: TIỀN XỬ LÝ DỮ LIỆU")
print("=" * 60)

# Đọc dữ liệu
train = pd.read_csv("data/train_df.csv")
val = pd.read_csv("data/val_df.csv")
test = pd.read_csv("data/test_df.csv")

# =====================================================
# 2.2.1 THỐNG KÊ SỐ LƯỢNG DỮ LIỆU
# =====================================================

print("\n2.2.1 THỐNG KÊ SỐ LƯỢNG DỮ LIỆU")
print("-" * 60)

print("Train:", train.shape)
print("Validation:", val.shape)
print("Test:", test.shape)

total_samples = len(train) + len(val) + len(test)

print("\nTổng số mẫu:", total_samples)

# =====================================================
# PHÂN BỐ NHÃN
# =====================================================

print("\nPHÂN BỐ NHÃN TRAIN")
print("-" * 60)

print(train["labels"].value_counts())

# =====================================================
# 2.2.2 KHẢO SÁT ĐẶC ĐIỂM DỮ LIỆU
# =====================================================

print("\n2.2.2 KHẢO SÁT ĐẶC ĐIỂM DỮ LIỆU")
print("-" * 60)

train["length"] = train["cmt_col"].astype(str).apply(len)

print(train["length"].describe())

# =====================================================
# HIỂN THỊ MỘT SỐ BÌNH LUẬN MẪU
# =====================================================

print("\n20 BÌNH LUẬN MẪU")
print("-" * 60)

for i, comment in enumerate(train["cmt_col"].sample(20)):
    print(f"{i+1}. {comment}")

# =====================================================
# 2.3 TIỀN XỬ LÝ DỮ LIỆU
# =====================================================

# -----------------------------------------------------
# 2.3.1 Loại bỏ URL
# -----------------------------------------------------

def remove_url(text):

    return re.sub(
        r"http\S+|www\S+",
        "",
        str(text)
    )


# -----------------------------------------------------
# 2.3.2 Loại bỏ hashtag và username
# -----------------------------------------------------

def remove_hashtag_username(text):

    text = re.sub(r"#\w+", "", text)

    text = re.sub(r"@\w+", "", text)

    return text


# -----------------------------------------------------
# 2.3.3 Chuyển về chữ thường
# -----------------------------------------------------

def lowercase(text):

    return text.lower()


# -----------------------------------------------------
# 2.3.4 Chuẩn hóa khoảng trắng
# -----------------------------------------------------

def normalize_space(text):

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# -----------------------------------------------------
# 2.3.5 Chuẩn hóa teencode và từ viết tắt
# -----------------------------------------------------

teencode_dict = {
    "ko": "không",
    "k": "không",
    "hok": "không",
    "khum": "không",
    "dc": "được",
    "đc": "được",
    "ntn": "như thế nào",
    "j": "gì",
    "cx": "cũng",
    "vs": "với",
    "ns": "nói",
    "ib": "inbox",
    "ad": "admin",
    "mn": "mọi người",
    "mik": "mình",
    "mk": "mình",
    "bn": "bạn",
    "ae": "anh em",
    "bt": "bình thường"
}

def normalize_teencode(text):

    words = text.split()

    new_words = []

    for word in words:

        if word in teencode_dict:
            new_words.append(teencode_dict[word])
        else:
            new_words.append(word)

    return " ".join(new_words)


# -----------------------------------------------------
# 2.3.6 Tách từ tiếng Việt bằng PyVi
# -----------------------------------------------------

def word_segmentation(text):

    return ViTokenizer.tokenize(text)


# -----------------------------------------------------
# Pipeline tiền xử lý
# -----------------------------------------------------

def preprocess_text(text):

    text = remove_url(text)

    text = remove_hashtag_username(text)

    text = lowercase(text)

    text = normalize_space(text)

    text = normalize_teencode(text)

    text = word_segmentation(text)

    return text


# =====================================================
# THỰC HIỆN TIỀN XỬ LÝ
# =====================================================

print("\nĐANG TIỀN XỬ LÝ DỮ LIỆU...")
print("-" * 60)

train["processed_text"] = train["cmt_col"].apply(preprocess_text)
val["processed_text"] = val["cmt_col"].apply(preprocess_text)
test["processed_text"] = test["cmt_col"].apply(preprocess_text)

print("HOÀN THÀNH!")

# =====================================================
# 2.4 KẾT QUẢ SAU TIỀN XỬ LÝ
# =====================================================

print("\n2.4 KẾT QUẢ SAU TIỀN XỬ LÝ")
print("-" * 60)

for i in range(5):

    print("\nBÌNH LUẬN GỐC:")
    print(train["cmt_col"].iloc[i])

    print("\nSAU TIỀN XỬ LÝ:")
    print(train["processed_text"].iloc[i])

    print("-" * 60)

# =====================================================
# 2.3.7 TOKENIZATION BẰNG PHOBERT TOKENIZER
# =====================================================

print("\nTOKENIZATION BẰNG PHOBERT")
print("-" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    "vinai/phobert-base"
)

example = train["processed_text"].iloc[0]

tokens = tokenizer.tokenize(example)

print("Văn bản:")
print(example)

print("\nTokens:")
print(tokens)

# =====================================================
# LƯU DỮ LIỆU SAU TIỀN XỬ LÝ
# =====================================================

train.to_csv(
    "output/train_processed.csv",
    index=False,
    encoding="utf-8-sig"
)

val.to_csv(
    "output/val_processed.csv",
    index=False,
    encoding="utf-8-sig"
)

test.to_csv(
    "output/test_processed.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nĐÃ LƯU DỮ LIỆU THÀNH CÔNG")
print("output/train_processed.csv")
print("output/val_processed.csv")
print("output/test_processed.csv")