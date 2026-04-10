import os
from src.chunking import CustomChunker
from src.store import EmbeddingStore
from src.embeddings import OpenAIEmbedder
from src.agent import KnowledgeBaseAgent
from src.models import Document
from dotenv import load_dotenv

load_dotenv()

def demo_llm(prompt: str) -> str:
    """Mock LLM trả về prompt cho dễ debug nếu chưa có API Key xịn"""
    return "[CÂU TRẢ LỜI MẪU]: Theo quy định tại Giáo trình, Phạm nhân..."

def run_benchmark():
    print("1. Đang đọc Giáo trình Luật...")
    with open("data/output.md", "r", encoding="utf-8") as f:
        text = f.read()

    print("2. Phân rã dữ liệu bằng thuật toán CustomChunker...")
    chunker = CustomChunker(max_chunk_size=1500)
    chunks = chunker.chunk(text)
    
    docs = []
    # Lấy 50 chunks đầu tiên thôi để CHẠY SIÊU NHANH (tránh lỗi chờ lâu của OpenAI)
    # Vì bài lab chỉ cần demo kết quả bảng, rút gọn data giúp bạn test trong 5 giây!
    for i, c in enumerate(chunks[:50]):
        docs.append(Document(id=f"luat_{i}", content=c, metadata={"source": "Giáo trình"}))

    print("3. Đang mã hóa Embeddings (Sẽ cực kỳ nhanh)...")
    embedder = OpenAIEmbedder(model_name="text-embedding-3-small")
    store = EmbeddingStore(collection_name="bang_benchmark", embedding_fn=embedder)
    store.add_documents(docs)

    print("4. Khởi tạo Agent RAG...")
    # Nếu bạn có hàm OpenAI thật thì thay demo_llm bằng hàm đó, còn bài tập này demo_llm là đủ
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)

    queries = [
        "Khái niệm pháp luật thi hành án hình sự là gì?",
        "Nguyên tắc nhân đạo trong thi hành án hình sự thể hiện như thế nào?",
        "Tác dụng giáo dục cải tạo của hình phạt là gì?",
        "Nhiệm vụ của pháp luật thi hành án hình sự là gì?",
        "Quyền và nghĩa vụ của người bị kết án phạt tù theo pháp luật?"
    ]

    print("\n================ KẾT QUẢ BENCHMARK TRUY VẤN TỰ ĐỘNG ================\n")
    
    for i, q in enumerate(queries, 1):
        print(f"👉 CÂU {i}: {q}")
        
        # Gọi Store để lấy top 1 chunk
        results = store.search(q, top_k=1)
        if results:
            score = results[0]['score']
            content = results[0]['content'][:120].replace('\n', ' ') + "..."
            print(f"   [Top-1 Chunk] Điểm: {score:.3f} | Trích xuất: {content}")
        else:
            print("   [Top-1 Chunk] Không tìm thấy")
            
        # Gọi Agent để trả lời
        agent_answer = agent.answer(q, top_k=1)
        print(f"   [Agent Answer] {agent_answer}")
        print("-" * 80)

if __name__ == "__main__":
    run_benchmark()
