import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# 1. Load model (hỗ trợ tiếng Việt)
print("Loading model...")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
print("Model loaded!\n")

# 2. Dữ liệu pairs
pairs = [
    ("Người bị kết án tự nguyện bồi thường", "Phạm nhân tự giác khắc phục hậu quả"),
    ("Con chó đang nằm ngủ trên nệm", "Chú cún đang đánh giấc trên giường"),
    ("Tôi rất yêu màu xanh bầu trời", "Tôi rất ghét thiên nhiên"), 
    ("Mức án tử hình", "Nấu ăn trong trại tạm giam"),
    ("Hãng Apple ra mắt điện thoại mới", "Tôi vừa mua một quả táo ngon"),
]

# 3. Tính similarity cho từng pair
print("=" * 80)
print("KẾT QUẢ TÍNH ACTUAL SCORE")
print("=" * 80)

results = []

for i, (sent_a, sent_b) in enumerate(pairs, 1):
    # Encode từng câu
    emb_a = model.encode(sent_a, convert_to_numpy=True)
    emb_b = model.encode(sent_b, convert_to_numpy=True)
    
    # Chuẩn hóa vector (nếu chưa được chuẩn hóa)
    emb_a_norm = emb_a / np.linalg.norm(emb_a)
    emb_b_norm = emb_b / np.linalg.norm(emb_b)
    
    # Cosine similarity = dot product của 2 vector đã chuẩn hóa
    similarity = np.dot(emb_a_norm, emb_b_norm)
    
    # Hoặc dùng hàm có sẵn:
    # similarity = util.cos_sim(emb_a, emb_b)[0][0].item()
    
    results.append({
        "Pair": i,
        "Sentence A": sent_a[:50] + "..." if len(sent_a) > 50 else sent_a,
        "Sentence B": sent_b[:50] + "..." if len(sent_b) > 50 else sent_b,
        "Actual Score": round(similarity, 4)
    })
    
    print(f"\nPair {i}:")
    print(f"  A: {sent_a}")
    print(f"  B: {sent_b}")
    print(f"  Actual Score: {similarity:.4f}")

# 4. Xuất ra bảng đẹp
print("\n" + "=" * 80)
print("BẢNG TỔNG HỢP")
print("=" * 80)

df = pd.DataFrame(results)
print(df.to_string(index=False))

# 5. Thống kê
print("\n" + "=" * 80)
print("THỐNG KÊ")
print("=" * 80)
print(f"Score cao nhất: {max(r['Actual Score'] for r in results):.4f} (Pair {np.argmax([r['Actual Score'] for r in results]) + 1})")
print(f"Score thấp nhất: {min(r['Actual Score'] for r in results):.4f} (Pair {np.argmin([r['Actual Score'] for r in results]) + 1})")
print(f"Score trung bình: {np.mean([r['Actual Score'] for r in results]):.4f}")

# 6. So sánh với dự đoán (giả sử bạn đã có dự đoán)
print("\n" + "=" * 80)
print("SO SÁNH VỚI DỰ ĐOÁN (giả định)")
print("=" * 80)

# Nhập dự đoán của bạn
predictions = ["high", "high", "low", "high", "low"]  # Sửa theo dự đoán thực tế của bạn
for i, (pred, actual) in enumerate(zip(predictions, [r['Actual Score'] for r in results]), 1):
    pred_binary = 1 if pred == "high" else 0
    actual_binary = 1 if actual > 0.5 else 0
    is_correct = "✅" if pred_binary == actual_binary else "❌"
    print(f"Pair {i}: Dự đoán={pred:4} | Actual={actual:.4f} | {'Đúng' if pred_binary == actual_binary else 'Sai'} {is_correct}")

# 7. Lưu ra file CSV
df.to_csv("similarity_scores.csv", index=False, encoding='utf-8-sig')
print("\n✅ Đã lưu kết quả vào file 'similarity_scores.csv'")