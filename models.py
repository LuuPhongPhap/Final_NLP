import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. TEXT CNN (KimCNN)
# Ý tưởng: Dùng các bộ lọc (kernels) kích thước khác nhau để "bắt" các cụm từ (n-grams) quan trọng.
# ==========================================
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=[3, 4, 5], num_filters=100, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Tạo nhiều lớp Convolution 1D song song với các kernel_size khác nhau
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k) 
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x) # (batch_size, seq_len, embed_dim)
        x = x.permute(0, 2, 1) # Đổi chiều cho Conv1d: (batch_size, embed_dim, seq_len)
        
        # Chạy qua Conv -> ReLU -> MaxPool cho từng kernel size
        conv_results = []
        for conv in self.convs:
            c = F.relu(conv(x))
            pooled = F.max_pool1d(c, c.size(2)).squeeze(2) # (batch_size, num_filters)
            conv_results.append(pooled)
            
        # Nối các đặc trưng lại với nhau
        x = torch.cat(conv_results, dim=1) # (batch_size, len(kernel_sizes) * num_filters)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

# ==========================================
# 2. BiLSTM + ATTENTION
# Ý tưởng: BiLSTM đọc hai chiều, Attention giúp mô hình tập trung vào những từ quan trọng nhất trong câu.
# ==========================================
class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5):
        super(BiLSTMAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                            bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Cơ chế Attention
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x) # (batch_size, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded) # (batch_size, seq_len, hidden_dim * 2)
        
        # Tính điểm Attention
        attn_scores = self.attention_weights(lstm_out).squeeze(-1) # (batch_size, seq_len)
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1) # (batch_size, seq_len, 1)
        
        # Nhân trọng số Attention với output của LSTM
        context_vector = torch.sum(lstm_out * attn_weights, dim=1) # (batch_size, hidden_dim * 2)
        
        out = self.dropout(context_vector)
        logits = self.fc(out)
        return logits

# ==========================================
# 3. TEXT RCNN (Recurrent Convolutional Neural Network)
# Ý tưởng: Kết hợp Embedding gốc và ngữ cảnh từ BiLSTM, sau đó dùng Max Pooling như CNN.
# ==========================================
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
        embedded = self.embedding(x) # (batch_size, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded) # (batch_size, seq_len, hidden_dim * 2)
        
        # Nối (Concat) Embedding ban đầu và Ngữ cảnh từ LSTM
        x = torch.cat((embedded, lstm_out), dim=2) # (batch_size, seq_len, embed_dim + hidden_dim*2)
        
        # Pass qua Linear + Tanh
        y = torch.tanh(self.W2(x)) # (batch_size, seq_len, hidden_dim)
        y = y.permute(0, 2, 1) # Đổi chiều để MaxPool1d
        
        # Max Pooling over time (Lấy đặc trưng nổi bật nhất)
        z = F.max_pool1d(y, y.size(2)).squeeze(-1) # (batch_size, hidden_dim)
        z = self.dropout(z)
        logits = self.fc(z)
        return logits

# ==========================================
# 4. CUSTOM TRANSFORMER ENCODER (2 Layers)
# Ý tưởng: Tự xây dựng khối Transformer Encoder từ đầu, thêm Positional Encoding như mô hình gốc.
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, num_heads=8, num_layers=2, hidden_dim=512, dropout=0.5):
        super(CustomTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # Khối Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                   dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len)
        mask = (x == 0) # Tạo mask để bỏ qua các token padding (giả sử pad_token_id = 0)
        
        embedded = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        embedded = self.pos_encoder(embedded)
        
        # Pass qua Transformer
        transformer_out = self.transformer_encoder(embedded, src_key_padding_mask=mask)
        
        # Average Pooling (Chỉ lấy trung bình những token KHÔNG phải padding)
        mask_expanded = mask.unsqueeze(-1).expand_as(transformer_out)
        transformer_out = transformer_out.masked_fill(mask_expanded, 0.0)
        sum_embeddings = torch.sum(transformer_out, dim=1)
        sum_mask = (~mask).float().sum(dim=1, keepdim=True)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled = sum_embeddings / sum_mask # (batch_size, embed_dim)
        
        out = self.dropout(pooled)
        logits = self.fc(out)
        return logits

# --- KIỂM TRA THỬ KHỞI TẠO MÔ HÌNH ---
if __name__ == "__main__":
    VOCAB_SIZE = 30000  # Kích thước từ vựng
    EMBED_DIM = 256     # Số chiều vector từ
    NUM_CLASSES = 3     # 3 nhãn: Cơ khí, Phần mềm, Con người
    
    # Khởi tạo thử 4 model
    m1 = TextCNN(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES)
    m2 = BiLSTMAttention(VOCAB_SIZE, EMBED_DIM, hidden_dim=128, num_classes=NUM_CLASSES)
    m3 = TextRCNN(VOCAB_SIZE, EMBED_DIM, hidden_dim=128, num_classes=NUM_CLASSES)
    m4 = CustomTransformer(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES, num_heads=8, num_layers=2)
    
    # In ra số lượng tham số (Theo yêu cầu của file PDF: báo cáo "Số tham số ước lượng")
    print(f"Tham số TextCNN: {sum(p.numel() for p in m1.parameters() if p.requires_grad):,}")
    print(f"Tham số BiLSTM+Attn: {sum(p.numel() for p in m2.parameters() if p.requires_grad):,}")
    print(f"Tham số TextRCNN: {sum(p.numel() for p in m3.parameters() if p.requires_grad):,}")
    print(f"Tham số CustomTransformer: {sum(p.numel() for p in m4.parameters() if p.requires_grad):,}")