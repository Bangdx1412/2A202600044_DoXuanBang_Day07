from src.chunking import compute_similarity
from src.embeddings import OpenAIEmbedder
from dotenv import load_dotenv

load_dotenv()

embedder = OpenAIEmbedder(model_name="text-embedding-3-small")

pairs = [
    ("Người bị kết án tự nguyện bồi thường", "Phạm nhân tự giác khắc phục hậu quả", "high"),
    ("Con chó đang nằm ngủ trên nệm", "Chú cún đang đánh giấc trên giường", "high"),
    ("Tôi rất yêu màu xanh bầu trời", "Tôi rất ghét thiên nhiên", "low"), 
    ("Mức án tử hình", "Nấu ăn trong trại tạm giam", "low"),
    ("Hãng Apple ra mắt điện thoại mới", "Tôi vừa mua một quả táo ngon", "low"),
]

for a, b, pred in pairs:
    v1 = embedder(a)
    v2 = embedder(b)
    score = compute_similarity(v1, v2)
    print(f"{a} | {b} | Pred: {pred} | Score: {score:.3f}")
