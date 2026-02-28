import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

# ==========================================
# BƯỚC 1 & 2: ĐỌC DỮ LIỆU VÀ TOKENIZATION
# ==========================================
print("1. Đang tải dữ liệu và Tokenizer...")
df = pd.read_csv('nlp_dataset_combined.csv')
df['report_text'] = df['report_text'].astype(str)
df['label'] = df['label'].astype(int)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["report_text"], padding="max_length", truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# ==========================================
# BƯỚC 3: KHỞI TẠO MÔ HÌNH
# ==========================================
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "f1_score": f1}

# ==========================================
# BƯỚC 4: HUẤN LUYỆN MÔ HÌNH
# ==========================================
print("2. Bắt đầu quá trình Fine-tuning...")
training_args = TrainingArguments(
    output_dir="./incident_model_results",
    learning_rate=3e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",       # Đánh giá sau mỗi epoch
    save_strategy="epoch",
    logging_strategy="epoch",    # Bắt buộc log lại để lát vẽ đồ thị
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# ==========================================
# BƯỚC 5: VẼ ĐỒ THỊ HUẤN LUYỆN (TRAINING CURVES)
# ==========================================
print("\n3. Đang vẽ đồ thị đánh giá...")
log_history = trainer.state.log_history

# Trích xuất dữ liệu từ log
epochs = []
train_loss = []
eval_loss = []
eval_acc = []
eval_f1 = []

for entry in log_history:
    if 'loss' in entry and 'epoch' in entry: # Training loss
        epochs.append(entry['epoch'])
        train_loss.append(entry['loss'])
    elif 'eval_loss' in entry: # Evaluation metrics
        eval_loss.append(entry['eval_loss'])
        eval_acc.append(entry['eval_accuracy'])
        eval_f1.append(entry['eval_f1_score'])

# Đảm bảo số lượng epochs khớp nhau để vẽ đồ thị
epochs = list(range(1, len(eval_loss) + 1))

# -- Hình 1: Đồ thị Loss (Train vs Eval) --
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
# Nếu train_loss nhiều hơn eval_loss, ta nội suy hoặc chỉ lấy điểm cuối mỗi epoch
plt.plot(epochs, eval_loss, label='Validation Loss', color='red', marker='o')
if len(train_loss) == len(eval_loss):
    plt.plot(epochs, train_loss, label='Training Loss', color='blue', marker='o')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# -- Hình 2: Đồ thị Accuracy & F1-Score --
plt.subplot(1, 2, 2)
plt.plot(epochs, eval_acc, label='Validation Accuracy', color='green', marker='s')
plt.plot(epochs, eval_f1, label='Validation F1-Score', color='purple', marker='^')
plt.title('Accuracy & F1-Score over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show() # Hiển thị đồ thị cho bạn xem

# -- Ma trận nhầm lẫn (Confusion Matrix) --
predictions = trainer.predict(tokenized_test)
preds = np.argmax(predictions.predictions, axis=-1)
y_true = predictions.label_ids
target_names = ['Mechanical (0)', 'Software (1)', 'Human (2)']

cm = confusion_matrix(y_true, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Thực tế (True Label)')
plt.xlabel('Dự đoán (Predicted Label)')
plt.title('Ma trận Nhầm lẫn (Confusion Matrix)')
plt.show()

# ==========================================
# BƯỚC 6: NHẬP TRỰC TIẾP CÂU ĐỂ TEST
# ==========================================
trainer.save_model("./final_vehicle_incident_model")

def predict_incident(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(trainer.model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = trainer.model(**inputs)
        
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class_id].item()
    
    labels_dict = {0: 'Cơ khí (Mechanical)', 1: 'Phần mềm (Software)', 2: 'Con người (Human)'}
    print(f"\n=> 🤖 MÔ HÌNH DỰ ĐOÁN LỖI: {labels_dict[predicted_class_id]}")
    print(f"=> 🎯 ĐỘ TIN CẬY (Confidence): {confidence*100:.2f}%\n")

print("\n" + "="*50)
print("🚀 HỆ THỐNG PHÂN LOẠI SỰ CỐ ĐÃ SẴN SÀNG!")
print("="*50)

# Vòng lặp cho phép người dùng nhập văn bản liên tục
while True:
    user_input = input("✍️ Nhập câu mô tả sự cố (hoặc gõ 'q' để thoát): ")
    if user_input.lower() in ['q', 'quit', 'exit', 'thoat']:
        print("Đã thoát chương trình.")
        break
    if user_input.strip() == "":
        continue
    
    predict_incident(user_input)