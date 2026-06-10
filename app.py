import streamlit as st
import joblib
import pandas as pd
import torch
import os
from pyvi import ViTokenizer

# 1. Cấu hình trang
st.set_page_config(page_title="AI Moderator Pro", page_icon="🛡️", layout="wide")

# 2. HÀM TẢI MÔ HÌNH TỪ FILE PKL 
def load_models_from_pkl():
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Đọc chính xác 3 file .pkl trong thư mục
    vec_path = os.path.join(base_path, 'tfidf_vectorizer.pkl')
    lr_path = os.path.join(base_path, 'logreg_model.pkl')
    svm_path = os.path.join(base_path, 'svm_model.pkl')
    
    # Kiểm tra xem file có tồn tại không
    if not os.path.exists(svm_path):
        raise FileNotFoundError("Không tìm thấy file svm_model.pkl! Hãy chắc chắn bạn đã train và lưu file vào đúng thư mục này.")

    vec = joblib.load(vec_path)
    lr = joblib.load(lr_path)
    svm = joblib.load(svm_path)
    
    pb_tokenizer = None
    pb_model = None
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model_path = os.path.join(base_path, "phobert_final_model")
        if os.path.exists(model_path):
            pb_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            pb_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except Exception:
        pass 
        
    return vec, lr, svm, pb_tokenizer, pb_model

try:
    vectorizer, logreg, svm, pb_tokenizer, pb_model = load_models_from_pkl()
except Exception as e:
    st.error(f"Lỗi khi đọc file .pkl: {e}")
    st.stop()

# ==========================================
# THANH BÊN (SIDEBAR)
# ==========================================
with st.sidebar:
    st.title("Cấu hình AI")
    
    options = ["Logistic Regression", "SVM (Linear)"]
    if pb_model is not None:
        options.append("PhoBERT (Deep Learning)")
    else:
        st.warning("⚠️ Không tìm thấy PhoBERT, tính năng này đã bị ẩn.")
        
    model_choice = st.radio("Chọn mô hình phân tích:", options)
    st.markdown("---")

# ==========================================
# MÀN HÌNH CHÍNH
# ==========================================
st.title("Hệ thống Kiểm duyệt Bình luận tự động")
st.write("Dự án Trí tuệ Nhân tạo - Phân loại: **CLEAN / OFFENSIVE / HATE**")

col1, col2 = st.columns([2, 1.2])

with col1:
    st.subheader("Nhập nội dung")
    user_input = st.text_area("Bình luận của người dùng:", height=150, placeholder="Nhập câu cần kiểm tra...")
    analyze_btn = st.button("Phân tích ngay", type="primary", use_container_width=True)

with col2:
    st.subheader("Kết quả phân tích")
    
    if analyze_btn and user_input.strip():
        with st.spinner("AI đang đọc file và suy nghĩ..."):
            
            # --- XỬ LÝ DỰ ĐOÁN ---
            if model_choice == "PhoBERT (Deep Learning)":
                text_segmented = ViTokenizer.tokenize(user_input)
                inputs = pb_tokenizer(text_segmented, return_tensors="pt", truncation=True, max_length=256)
                with torch.no_grad():
                    outputs = pb_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probs, dim=-1).item()
                confidences = probs[0].tolist()
            else:
                model = logreg if model_choice == "Logistic Regression" else svm
                text_vec = vectorizer.transform([user_input])
                prediction = model.predict(text_vec)[0]
                confidences = model.predict_proba(text_vec)[0]

            # --- HIỂN THỊ ---
            labels = ["BÌNH THƯỜNG (CLEAN)", "XÚC PHẠM (OFFENSIVE)", "THÙ GHÉT (HATE)"]
            icons = ["✅", "⚠️", "🚨"]
            
            p = int(prediction)
            conf = confidences[p] * 100
            
            if p == 0: st.success(f"{icons[p]} {labels[p]}")
            elif p == 1: st.warning(f"{icons[p]} {labels[p]}")
            else: st.error(f"{icons[p]} {labels[p]}")
            
            st.metric("Độ tự tin của AI", f"{conf:.2f}%")
            st.progress(float(conf) / 100.0)
            
            with st.expander("Xem chi tiết xác suất"):
                for i, label in enumerate(labels):
                    st.write(f"{label}: {confidences[i]*100:.1f}%")
    elif analyze_btn:
        st.warning("Vui lòng không để trống nội dung!")