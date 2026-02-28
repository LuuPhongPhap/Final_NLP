import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, set_seed
from datasets import Dataset

# ==========================================
# CÀI ĐẶT CHUNG
# ==========================================
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["report_text"], padding="max_length", truncation=True, max_length=128)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "macro_f1": macro_f1}

print("Đang tải dữ liệu...")
# Lưu ý: File data trong Main_Bert.py của bạn tên là 'nlp_dataset_combined.csv'
df = pd.read_csv('nlp_dataset_combined.csv')
df['report_text'] = df['report_text'].astype(str)
df['label'] = df['label'].astype(int)

# ==========================================
# THỰC NGHIỆM 1: CROSS-DOMAIN
# Train trên tập NHTSA -> Test trên tập Tesla
# ==========================================
print("\n" + "="*50)
print("🚀 THỰC NGHIỆM 1: CROSS-DOMAIN")
print("="*50)

# Tách dữ liệu theo Domain (Nếu trong data của bạn có cột 'source')
if 'source' in df.columns:
    df_train_cd = df[df['source'].str.contains('NHTSA', case=False, na=False)]
    df_test_cd = df[df['source'].str.contains('Tesla', case=False, na=False)]

    if len(df_train_cd) > 0 and len(df_test_cd) > 0:
        print(f"Số lượng Train (NHTSA): {len(df_train_cd)} | Số lượng Test (Tesla): {len(df_test_cd)}")
        
        train_dataset_cd = Dataset.from_pandas(df_train_cd).map(tokenize_function, batched=True)
        test_dataset_cd = Dataset.from_pandas(df_test_cd).map(tokenize_function, batched=True)

        set_seed(42) # Cố định seed
        model_cd = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

        training_args_cd = TrainingArguments(
            output_dir="./results_cross_domain",
            learning_rate=3e-4, # Giữ nguyên Learning Rate như file cũ của bạn
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3, # 3 Epochs là đủ cho BERT để tránh overfitting
            report_to="none",
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True
        )

        trainer_cd = Trainer(
            model=model_cd,
            args=training_args_cd,
            train_dataset=train_dataset_cd,
            eval_dataset=test_dataset_cd,
            compute_metrics=compute_metrics,
        )

        print("Đang huấn luyện Cross-Domain...")
        trainer_cd.train()
        
        print("\nĐang đánh giá Cross-Domain trên tập Tesla...")
        results_cd = trainer_cd.evaluate()
        print(f"✅ Kết quả Cross-Domain -> Accuracy: {results_cd['eval_accuracy']:.4f} | Macro-F1: {results_cd['eval_macro_f1']:.4f}")
        
        # LƯU MÔ HÌNH CROSS-DOMAIN CHẮC CHẮN VÀO Ổ CỨNG
        trainer_cd.save_model("./saved_model_cross_domain")
        print("💾 Đã lưu mô hình Cross-Domain tại thư mục: './saved_model_cross_domain'")
    else:
        print("Không đủ dữ liệu cho 2 domain. Vui lòng kiểm tra lại cột 'source'.")
else:
    print("Không tìm thấy cột 'source' trong dữ liệu để làm Cross-Domain.")


# ==========================================
# THỰC NGHIỆM 2: 3 RANDOM SEEDS (TÍNH ỔN ĐỊNH)
# ==========================================
print("\n" + "="*50)
print("🚀 THỰC NGHIỆM 2: KIỂM TRA ĐỘ ỔN ĐỊNH VỚI 3 RANDOM SEEDS")
print("="*50)

# Chia tập 80/20 tiêu chuẩn cho toàn bộ data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
train_dataset_rs = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
test_dataset_rs = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

seeds = [42, 123, 999]
f1_scores = []
acc_scores = []

for seed in seeds:
    print(f"\n--- Đang chạy với Seed = {seed} ---")
    set_seed(seed) # Khởi tạo lại hệ thống sinh số ngẫu nhiên
    
    # Bắt buộc: Khởi tạo lại một bộ não mới tinh (chưa học gì) cho mỗi lần chạy
    model_rs = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    training_args_rs = TrainingArguments(
        output_dir=f"./results_seed_{seed}",
        learning_rate=3e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        report_to="none",
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True
    )

    trainer_rs = Trainer(
        model=model_rs,
        args=training_args_rs,
        train_dataset=train_dataset_rs,
        eval_dataset=test_dataset_rs,
        compute_metrics=compute_metrics,
    )

    trainer_rs.train()
    results_rs = trainer_rs.evaluate()
    
    acc = results_rs['eval_accuracy']
    macro_f1 = results_rs['eval_macro_f1']
    
    f1_scores.append(macro_f1)
    acc_scores.append(acc)
    print(f"-> Seed {seed} Hoàn thành: Macro-F1 = {macro_f1:.4f}, Accuracy = {acc:.4f}")
    
    # LƯU LẠI MÔ HÌNH CỦA TỪNG SEED
    save_path = f"./saved_model_seed_{seed}"
    trainer_rs.save_model(save_path)
    print(f"💾 Đã lưu mô hình Seed {seed} tại thư mục: '{save_path}'")

# ==========================================
# TỔNG KẾT VÀ BÁO CÁO KẾT QUẢ CHO WORD
# ==========================================
f1_mean = np.mean(f1_scores)
f1_std = np.std(f1_scores)

acc_mean = np.mean(acc_scores)
acc_std = np.std(acc_scores)

print("\n🏆 KẾT QUẢ CUỐI CÙNG CHO BÁO CÁO (MEAN ± STD):")
print(f"🔹 DistilBERT Macro-F1: {f1_mean:.4f} ± {f1_std:.4f}")
print(f"🔹 DistilBERT Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
print("\n=> Chú ý: Bạn hãy copy 2 dòng Mean ± Std này vào báo cáo. Toàn bộ mô hình đã được lưu an toàn trong thư mục code của bạn!")