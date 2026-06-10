import pandas as pd
from sklearn.svm import SVC
import joblib
import os

base_path = os.path.dirname(os.path.abspath(__file__))

print("1. Đang đọc Vectorizer và Dữ liệu...")
# Lấy Vectorizer đã có sẵn để đảm bảo khớp dữ liệu
vectorizer = joblib.load(os.path.join(base_path, 'tfidf_vectorizer.pkl'))

# Đọc file train
train_file = os.path.join(base_path, 'train_processed.csv')
df = pd.read_csv(train_file, usecols=[0, 1], encoding='utf-8-sig')
df.columns = ['processed_text', 'labels']
df = df.dropna(subset=['labels'])
df['labels'] = df['labels'].astype(int)
df['processed_text'] = df['processed_text'].fillna("").astype(str)

print("2. Đang Vector hóa dữ liệu...")
X_train = vectorizer.transform(df['processed_text'])
y_train = df['labels']

print("3. Đang huấn luyện lại SVM (vài giây)...")
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train, y_train)

print("4. Đang lưu file svm_model.pkl mới...")
svm_path = os.path.join(base_path, 'svm_model.pkl')
joblib.dump(svm, svm_path)

print(f"✅ XONG! Đã đè thành công file SVM mới tại: {svm_path}")