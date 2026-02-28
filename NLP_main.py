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
import math

# Cấu hình thiết bị (Sử dụng GPU nếu có, ngược lại dùng CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

# ==========================================
# PHẦN 1: ĐỊNH NGHĨA 4 KIẾN TRÚC MÔ HÌNH
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

class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5):
        super(BiLSTMAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                            bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(self.embedding(x))
        attn_weights = F.softmax(self.attention_weights(lstm_out).squeeze(-1), dim=-1).unsqueeze(-1)
        context_vector = torch.sum(lstm_out * attn_weights, dim=1) 
        return self.fc(self.dropout(context_vector))

class TextRCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5):
        super(TextRCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                            bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.W2 = nn.Linear(embed_dim + 2 * hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        x = torch.cat((embedded, lstm_out), dim=2) 
        y = torch.tanh(self.W2(x)).permute(0, 2, 1) 
        z = F.max_pool1d(y, y.size(2)).squeeze(-1) 
        return self.fc(self.dropout(z))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, num_heads=4, num_layers=2, hidden_dim=256, dropout=0.5):
        super(CustomTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                   dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        mask = (x == 0)
        embedded = self.pos_encoder(self.embedding(x) * math.sqrt(self.embedding.embedding_dim))
        transformer_out = self.transformer_encoder(embedded, src_key_padding_mask=mask)
        mask_expanded = mask.unsqueeze(-1).expand_as(transformer_out)
        transformer_out = transformer_out.masked_fill(mask_expanded, 0.0)
        sum_mask = torch.clamp((~mask).float().sum(dim=1, keepdim=True), min=1e-9)
        pooled = torch.sum(transformer_out, dim=1) / sum_mask 
        return self.fc(self.dropout(pooled))

# ==========================================
# PHẦN 2: CHUẨN BỊ DỮ LIỆU & HUẤN LUYỆN
# ==========================================
if __name__ == "__main__":
    print("1. Đang tải và chuẩn bị dữ liệu...")
    df = pd.read_csv('nlp_dataset_combined.csv')

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    VOCAB_SIZE = tokenizer.vocab_size

    def encode_texts(texts, max_len=128):
        encoded = tokenizer(texts.tolist(), padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        return encoded['input_ids']

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    X_train, y_train = encode_texts(train_df['report_text']), torch.tensor(train_df['label'].values, dtype=torch.long)
    X_val, y_val = encode_texts(val_df['report_text']), torch.tensor(val_df['label'].values, dtype=torch.long)
    X_test, y_test = encode_texts(test_df['report_text']), torch.tensor(test_df['label'].values, dtype=torch.long)

    BATCH_SIZE = 32
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, model_name="Model"):
        print(f"\n🚀 Bắt đầu huấn luyện mô hình: {model_name}")
        model.to(device)
        best_model_wts = copy.deepcopy(model.state_dict())
        best_f1 = 0.0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                
            train_loss /= len(train_loader.dataset)
            
            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
            val_loss /= len(val_loader.dataset)
            val_f1 = f1_score(all_labels, all_preds, average='macro')
            
            print(f"Epoch {epoch+1}/{num_epochs} | Time: {time.time() - start_time:.0f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                
        print(f"🎯 Train xong {model_name}. Best Val Macro-F1: {best_f1:.4f}")
        model.load_state_dict(best_model_wts)
        return model

    def evaluate_model(model, test_loader, model_name="Model"):
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
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
        return {"Model": model_name, "Accuracy": acc, "Macro-F1": macro_f1, "Weighted-F1": weighted_f1}

    EMBED_DIM = 128
    NUM_CLASSES = 3
    NUM_EPOCHS = 10 
    LR = 0.001

    models_dict = {
        "TextCNN (KimCNN)": TextCNN(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES),
        "BiLSTM + Attention": BiLSTMAttention(VOCAB_SIZE, EMBED_DIM, hidden_dim=64, num_classes=NUM_CLASSES),
        "TextRCNN": TextRCNN(VOCAB_SIZE, EMBED_DIM, hidden_dim=64, num_classes=NUM_CLASSES),
        "Custom Transformer": CustomTransformer(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES, num_heads=4, num_layers=2)
    }

    results = []
    criterion = nn.CrossEntropyLoss()

    for name, model in models_dict.items():
        optimizer = optim.Adam(model.parameters(), lr=LR)
        trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS, model_name=name)
        metrics = evaluate_model(trained_model, test_loader, model_name=name)
        results.append(metrics)
        torch.save(trained_model.state_dict(), f"{name.replace(' ', '_')}_best.pt")

    print("\n🏆 BẢNG TỔNG HỢP KẾT QUẢ 4 MÔ HÌNH (ĐỂ ĐƯA VÀO BÁO CÁO):")
    print(pd.DataFrame(results).to_string(index=False))

    # ==========================================
    # PHẦN 3: VẼ ĐỒ THỊ SO SÁNH KẾT QUẢ
    # ==========================================
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("\n📈 Đang vẽ đồ thị so sánh...")
    
    # Chuyển kết quả thành DataFrame
    results_df = pd.DataFrame(results)
    
    # Đặt cột 'Model' làm Index để dễ vẽ biểu đồ nhóm (Grouped Bar Chart)
    results_df.set_index('Model', inplace=True)
    
    # Thiết lập kích thước và style cho biểu đồ
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    # Vẽ biểu đồ cột
    ax = results_df.plot(kind='bar', figsize=(12, 6), colormap='viridis', edgecolor='black')
    
    # Làm đẹp biểu đồ
    plt.title('SO SÁNH HIỆU SUẤT 4 MÔ HÌNH DEEP LEARNING (TEXT CLASSIFICATION)', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Kiến trúc Mô hình', fontsize=12, fontweight='bold')
    plt.ylabel('Điểm số (0.0 - 1.0)', fontsize=12, fontweight='bold')
    plt.ylim(0, 1.1) # Để trục y cao hơn 1 chút lấy chỗ trống cho Legend
    plt.xticks(rotation=15, ha='right') # Nghiêng chữ để không bị đè lên nhau
    
    # Đưa Legend (chú giải) ra góc gọn gàng
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Gắn con số (Text label) trực tiếp lên từng cột để báo cáo trông chuyên nghiệp hơn
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points',
                    fontsize=9)
        
    plt.tight_layout()
    
    # Lưu ảnh tự động để bạn copy thẳng vào báo cáo Word
    plt.savefig('Model_Comparison_Chart.png', dpi=300, bbox_inches='tight')
    print("✅ Đã lưu biểu đồ thành file ảnh: 'Model_Comparison_Chart.png'")
    
    # Hiển thị biểu đồ lên màn hình
    plt.show()