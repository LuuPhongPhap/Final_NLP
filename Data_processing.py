import pandas as pd
import re

# ==========================================
# PHẦN 1: ĐỌC VÀ XỬ LÝ DỮ LIỆU TESLA
# ==========================================
print("1. Đang đọc dữ liệu từ các file Tesla...")

df_tesla1 = pd.read_csv('dataset/Tesla Deaths - Deaths.csv')
df_tesla2 = pd.read_csv('dataset/Tesla Deaths - Deaths (3).csv')

# Chuẩn hóa tên cột
df_tesla1.columns = df_tesla1.columns.str.strip()
df_tesla2.columns = df_tesla2.columns.str.strip()

# Trích xuất văn bản
text_tesla1 = df_tesla1['Description'].dropna().astype(str)
text_tesla2 = df_tesla2['Description'].dropna().astype(str)

df_tesla = pd.DataFrame({'report_text': pd.concat([text_tesla1, text_tesla2], ignore_index=True)})
df_tesla['source'] = 'Tesla_Data'

# ==========================================
# PHẦN 2: ĐỌC DỮ LIỆU NHTSA (TỆP LỚN > 100MB)
# ==========================================
print("2. Đang đọc file dữ liệu khiếu nại NHTSA khổng lồ...")

try:
    # Thay đổi tên file dưới đây cho khớp với tên file bạn vừa giải nén (có thể là .csv hoặc .txt)
    # on_bad_lines='skip' giúp bỏ qua các dòng bị lỗi cấu trúc
    df_nhtsa_raw = pd.read_csv('dataset/COMPLAINTS_RECEIVED_2025-2026.txt', low_memory=False, on_bad_lines='skip')
    
    # Xóa khoảng trắng và viết hoa tên cột để dễ tìm
    df_nhtsa_raw.columns = df_nhtsa_raw.columns.str.strip().str.upper()
    
    # Tìm cột chứa văn bản (Thường NHTSA đặt là CDESCR, SUMMARY hoặc DESCRIPTION)
    text_col = None
    if 'CDESCR' in df_nhtsa_raw.columns:
        text_col = 'CDESCR'
    elif 'DESCRIPTION' in df_nhtsa_raw.columns:
        text_col = 'DESCRIPTION'
    elif 'SUMMARY' in df_nhtsa_raw.columns:
        text_col = 'SUMMARY'
    else:
        # Nếu không có tên chuẩn, tự động quét cột chứa đoạn văn dài nhất
        max_len = 0
        for col in df_nhtsa_raw.columns:
            avg_len = df_nhtsa_raw[col].astype(str).apply(len).mean()
            if avg_len > max_len:
                max_len = avg_len
                text_col = col

    print(f"   -> Đã tự động nhận diện cột văn bản NHTSA: '{text_col}'")
    
    # Rút trích cột văn bản, loại bỏ rỗng
    text_nhtsa = df_nhtsa_raw[text_col].dropna().astype(str)
    
    # TỐI ƯU RAM: Chỉ lấy mẫu ngẫu nhiên 50,000 dòng để không bị treo máy khi train mô hình
    n_samples = min(50000, len(text_nhtsa))
    text_nhtsa = text_nhtsa.sample(n=n_samples, random_state=42)
    
    df_nhtsa = pd.DataFrame({'report_text': text_nhtsa})
    df_nhtsa['source'] = 'NHTSA_Complaints_25_26'

except FileNotFoundError:
    print("\n❌ KHÔNG TÌM THẤY FILE NHTSA. Bạn đã giải nén và đổi tên đúng thành 'NHTSA_Complaints_25_26.csv' chưa?")
    df_nhtsa = pd.DataFrame(columns=['report_text', 'source'])

# ==========================================
# PHẦN 3: GỘP DỮ LIỆU VÀ GÁN NHÃN TỰ ĐỘNG
# ==========================================
print("3. Đang gộp dữ liệu và tiến hành phân tích từ khóa...")

# Gộp Tesla và NHTSA
df_final = pd.concat([df_tesla, df_nhtsa], ignore_index=True)
print(f"   -> Tổng số văn bản cần xử lý: {len(df_final)} dòng")

# Bộ từ khóa phân loại
keywords = {
    0: r"\b(brake|engine|tire|steering|transmission|mechanical|stall|pressure|rupture|locked|suspension|leak|battery|fire|burn)\b", # Cơ khí
    1: r"\b(autopilot|software|sensor|glitch|system|algorithm|automated|phantom|calibration|fsd|screen|reboot|navigate|autopark|camera|radar)\b", # Phần mềm
    2: r"\b(driver|texting|phone|asleep|drunk|speeding|distracted|intoxicated|panicked|fatigue|swerved|dui|unlicensed)\b"  # Con người
}

def assign_label_by_keyword(text):
    text = str(text).lower()
    scores = {0: 0, 1: 0, 2: 0}
    
    for label, pattern in keywords.items():
        matches = re.findall(pattern, text)
        scores[label] = len(matches)
    
    max_score = max(scores.values())
    if max_score == 0:
        return -1 
    
    top_labels = [label for label, score in scores.items() if score == max_score]
    return top_labels[0] if len(top_labels) == 1 else -1

# Áp dụng gán nhãn
df_final['label'] = df_final['report_text'].apply(assign_label_by_keyword)

label_map = {0: 'Mechanical', 1: 'Software', 2: 'Human', -1: 'Unknown'}
df_final['label_name'] = df_final['label'].map(label_map)

# ==========================================
# PHẦN 4: LÀM SẠCH VÀ XUẤT KẾT QUẢ
# ==========================================
print("4. Đang dọn dẹp và xuất file báo cáo...")

clean_data = df_final[df_final['label'] != -1].copy()
clean_data = clean_data.drop_duplicates(subset=['report_text'])

print("\n📊 --- KẾT QUẢ PHÂN LOẠI TOÀN BỘ DỮ LIỆU ---")
print(clean_data['label_name'].value_counts())
print("\n🔍 --- NGUỒN ĐÓNG GÓP ---")
print(clean_data.groupby(['label_name', 'source']).size().unstack(fill_value=0))

# Lưu file hoàn chỉnh
output_file = 'nlp_dataset_combined.csv'
clean_data.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✅ Đã hoàn tất! Dữ liệu sẵn sàng để train AI tại file: {output_file}")