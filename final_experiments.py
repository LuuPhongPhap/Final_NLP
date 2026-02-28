import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import time
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 0. ĐỊNH NGHĨA LẠI MÔ HÌNH TEXT-CNN (BEST MODEL)
# ==========================================
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=[3, 4, 5], num_filters=100, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k) 
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1) 
        conv_results = [F.max_pool1d(F.relu(conv(x)), conv(x).size(2)).squeeze(2) for conv in self.convs]
        x = torch.cat(conv_results, dim=1) 
        return self.fc(self.dropout(x))

# Hàm set_seed để đảm bảo kết quả có thể tái lập (Reproducibility)
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Hàm Training rút gọn
def train_and_eval(model, train_loader, test_loader, num_epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    
    # Train
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            
    # Eval
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, preds = torch.max(model(inputs), 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, macro_f1

# ==========================================
# CHUẨN BỊ DỮ LIỆU DÙNG CHUNG
# ==========================================
print("Đang chuẩn bị dữ liệu...")
df = pd.read_csv('nlp_dataset_combined.csv')
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
VOCAB_SIZE = tokenizer.vocab_size

def encode_texts(texts):
    return tokenizer(texts.tolist(), padding='max_length', truncation=True, max_length=128, return_tensors='pt')['input_ids']

print("\n" + "="*50)
print("🚀 THỰC NGHIỆM 1: CROSS-DOMAIN (Train: NHTSA -> Test: Tesla)")
print("="*50)
# Tách dữ liệu theo Nguồn (Domain)
# Giả sử trong cột 'source' của bạn có chứa từ khóa 'NHTSA' và 'Tesla'
df_train_domain = df[df['source'].str.contains('NHTSA', case=False, na=False)]
df_test_domain = df[df['source'].str.contains('Tesla', case=False, na=False)]

if len(df_train_domain) > 0 and len(df_test_domain) > 0:
    print(f"- Domain Train (NHTSA): {len(df_train_domain)} mẫu")
    print(f"- Domain Test (Tesla): {len(df_test_domain)} mẫu")
    
    X_train_cd = encode_texts(df_train_domain['report_text'])
    y_train_cd = torch.tensor(df_train_domain['label'].values, dtype=torch.long)
    X_test_cd = encode_texts(df_test_domain['report_text'])
    y_test_cd = torch.tensor(df_test_domain['label'].values, dtype=torch.long)
    
    train_loader_cd = DataLoader(TensorDataset(X_train_cd, y_train_cd), batch_size=32, shuffle=True)
    test_loader_cd = DataLoader(TensorDataset(X_test_cd, y_test_cd), batch_size=32, shuffle=False)
    
    set_seed(42)
    model_cd = TextCNN(VOCAB_SIZE, embed_dim=128, num_classes=3)
    acc_cd, f1_cd = train_and_eval(model_cd, train_loader_cd, test_loader_cd, num_epochs=4)
    print(f"✅ Kết quả Cross-Domain -> Accuracy: {acc_cd:.4f} | Macro-F1: {f1_cd:.4f}")
    print("=> Báo cáo: Kết quả này thường sẽ thấp hơn In-domain. Điều này chứng tỏ từ vựng giữa văn phong của NHTSA và văn phong của báo cáo Tesla có sự khác biệt (Domain Shift).")
else:
    print("Không tìm thấy đủ 2 domain trong cột 'source'. Vui lòng kiểm tra lại tên cột nguồn.")


print("\n" + "="*50)
print("🚀 THỰC NGHIỆM 2: KIỂM TRA ĐỘ ỔN ĐỊNH VỚI 3 RANDOM SEEDS")
print("="*50)

# Dùng toàn bộ data chia 80/20 tiêu chuẩn cho thực nghiệm này
X_all = encode_texts(df['report_text'])
y_all = torch.tensor(df['label'].values, dtype=torch.long)

X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
train_loader_rs = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
test_loader_rs = DataLoader(TensorDataset(X_te, y_te), batch_size=32, shuffle=False)

seeds = [42, 123, 999]
f1_scores = []
acc_scores = []

for seed in seeds:
    print(f"Đang chạy với Seed = {seed}...")
    set_seed(seed) # Khởi tạo lại trạng thái ngẫu nhiên
    
    # Phải khởi tạo LẠI mô hình hoàn toàn mới cho mỗi seed
    model_rs = TextCNN(VOCAB_SIZE, embed_dim=128, num_classes=3)
    
    acc, macro_f1 = train_and_eval(model_rs, train_loader_rs, test_loader_rs, num_epochs=4)
    f1_scores.append(macro_f1)
    acc_scores.append(acc)
    print(f"  -> Seed {seed}: Macro-F1 = {macro_f1:.4f}, Accuracy = {acc:.4f}")

# TÍNH TOÁN VÀ IN RA KẾT QUẢ ĐỂ COPY VÀO BÁO CÁO (MEAN ± STD)
f1_mean = np.mean(f1_scores)
f1_std = np.std(f1_scores)

acc_mean = np.mean(acc_scores)
acc_std = np.std(acc_scores)

print("\n🏆 KẾT QUẢ CUỐI CÙNG (COPY DÒNG NÀY VÀO BÁO CÁO WORD):")
print(f"🔹 Macro-F1: {f1_mean:.4f} ± {f1_std:.4f}")
print(f"🔹 Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")

if f1_std < 0.02:
    print("\n=> Đánh giá: Độ lệch chuẩn (Std) rất nhỏ (< 0.02), chứng tỏ mô hình của bạn HỘI TỤ RẤT ỔN ĐỊNH và KHÔNG BỊ PHỤ THUỘC VÀO TÍNH NGẪU NHIÊN của trọng số ban đầu.")