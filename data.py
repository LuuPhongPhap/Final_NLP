import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PretrainedConfig, TrainingArguments, Trainer
from datasets import Dataset
from transformers.modeling_outputs import SequenceClassifierOutput

# ==========================================
# BƯỚC 1: CHUẨN BỊ DỮ LIỆU & CHIA TẬP 80/10/10 
# ==========================================
print("1. Đang tải và chia tập dữ liệu...")
df = pd.read_csv('nlp_dataset_combined.csv')
df['report_text'] = df['report_text'].astype(str)
df['label'] = df['label'].astype(int)

# 1.1 Chia tập Train (80%) và tập Tạm (20%)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
# 1.2 Chia tập Tạm thành Validation (10%) và Test (10%)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

print(f"Kích thước tập dữ liệu: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

# 1.3 Xuất file để làm Inter-Annotator Agreement (Gán nhãn thủ công) [cite: 16]
# Trích ra 200 câu ngẫu nhiên để bạn và bạn bè gán nhãn tay tính Cohen's Kappa
iaa_sample = df.sample(n=200, random_state=99)[['report_text', 'label_name']]
iaa_sample['Người_gán_nhãn_1'] = ""
iaa_sample['Người_gán_nhãn_2'] = ""
iaa_sample.to_csv('IAA_Manual_Labeling_Task.csv', index=False, encoding='utf-8-sig')
print("-> Đã xuất file 'IAA_Manual_Labeling_Task.csv'. Bạn hãy nhờ 2 người điền vào file này để báo cáo nhé!")

# Chuyển sang định dạng Hugging Face
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["report_text"], padding="max_length", truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# ==========================================
# BƯỚC 2: TỰ THIẾT KẾ MÔ HÌNH VỚI CUSTOM HEAD 
# ==========================================
print("2. Đang khởi tạo mô hình DistilBERT với Custom Head (Multi-Sample Dropout)...")

class CustomDistilBertWithMSDrop(nn.Module):
    def __init__(self, model_name, num_labels):
        super(CustomDistilBertWithMSDrop, self).__init__()
        self.num_labels = num_labels
        # Load backbone
        self.distilbert = AutoModel.from_pretrained(model_name)
        # Tự thiết kế Head: Multi-Sample Dropout giúp mô hình chống Overfitting cực tốt
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)])
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        # Lấy hidden state của token đầu tiên [CLS]
        pooled_output = outputs[0][:, 0] 
        
        # Chạy qua 5 dropout layer khác nhau và lấy trung bình (Multi-Sample Dropout)
        logits = sum([self.classifier(dropout(pooled_output)) for dropout in self.dropouts]) / len(self.dropouts)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)

model = CustomDistilBertWithMSDrop(model_name, num_labels=3)

# ==========================================
# BƯỚC 3: METRICS ĐÁNH GIÁ ĐẦY ĐỦ NHẤT [cite: 53, 54, 55, 59]
# ==========================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions) # Accuracy
    macro_f1 = f1_score(labels, predictions, average='macro') # Macro-F1
    weighted_f1 = f1_score(labels, predictions, average='weighted') # Weighted-F1
    
    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}

# ==========================================
# BƯỚC 4: HUẤN LUYỆN (TRAINING)
# ==========================================
print("3. Bắt đầu huấn luyện...")
training_args = TrainingArguments(
    output_dir="./incident_model_results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val, # Dùng Validation set để đánh giá trong lúc train 
    compute_metrics=compute_metrics,
)

trainer.train()

# ==========================================
# BƯỚC 5: ĐÁNH GIÁ TRÊN TẬP TEST VÀ PHÂN TÍCH LỖI [cite: 85, 92]
# ==========================================
print("\n4. Đánh giá trên tập TEST và trích xuất lỗi...")
# Dự đoán trên tập Test
predictions = trainer.predict(tokenized_test)
preds = np.argmax(predictions.predictions, axis=-1)
y_true = predictions.label_ids
target_names = ['Mechanical (0)', 'Software (1)', 'Human (2)']

print("\n--- BÁO CÁO PHÂN LOẠI CHI TIẾT (Per-class Metrics) ---")
print(classification_report(y_true, preds, target_names=target_names))

# 1. Vẽ Ma trận nhầm lẫn CHUẨN HÓA THEO HÀNG (Row-normalized) 
cm_normalized = confusion_matrix(y_true, preds, normalize='true')
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Thực tế (True Label)')
plt.xlabel('Dự đoán (Predicted Label)')
plt.title('Ma trận Nhầm lẫn Chuẩn hóa (Normalized Confusion Matrix)')
plt.show()

# 2. TRÍCH XUẤT ERROR ANALYSIS (Tối thiểu 30 lỗi) 
test_df_results = test_df.copy()
test_df_results['predicted_label'] = preds

# Lọc ra những câu dự đoán sai
errors_df = test_df_results[test_df_results['label'] != test_df_results['predicted_label']]

# Lấy các mẫu lỗi (ưu tiên lấy khoảng 50 mẫu để bạn chọn ra 30 mẫu phân tích trong báo cáo)
sample_errors = errors_df.sample(n=min(50, len(errors_df)), random_state=42)
sample_errors.to_csv('Error_Analysis_Cases.csv', index=False, encoding='utf-8-sig')

print(f"\n✅ Đã tìm thấy {len(errors_df)} lỗi trên tập Test.")
print("=> Đã xuất 50 mẫu lỗi ra file 'Error_Analysis_Cases.csv'. Bạn hãy mở file này ra, chọn 30 câu để đưa vào phần PHÂN TÍCH LỖI (Error Analysis) trong báo cáo nhé!")