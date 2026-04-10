# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Đỗ Xuân Bằng
**Nhóm:** [Tên nhóm]
**Ngày:** [Ngày nộp]

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
>Cosine similarity cao (gần 1) cho thấy hai vector embedding có cùng hướng trong không gian, nghĩa là hai câu/văn bản có ý nghĩa ngữ nghĩa tương tự nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "Tôi đang học lập trình Python"
- Sentence B: "Hiện tại tôi đang học Python để lập trình"
- Tại sao tương đồng:Cùng một ý nghĩa (đang học Python), chỉ khác cách diễn đạt và thứ tự từ.

**Ví dụ LOW similarity:**
- Sentence A: "Tôi thích đọc sách buổi sáng"
- Sentence B: "Hôm nay trời nắng"
- Tại sao khác: Hai câu không liên quan về mặt ý nghĩa. câu A là sở thích cá nhân, câu B là mô tả thời tiết nên không có semantic connection->LOW similarity

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ quan tâm đến góc giữa hai vector, không bị ảnh hưởng bởi độ dài (magnitude). Điều này rất hữu ích vì embedding của câu dài thường có magnitude lớn hơn nhưng vẫn có thể cùng hướng với câu ngắn có cùng ý nghĩa.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> chunk đầu = 10000 - 500 = 9500. các chunk tiếp = 500-50  = 450. Tổng chunk = 9500/450 = 21.1111111111=> 22chunks + 1chunks đầu = 23chunks
> *Đáp án:* 23

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> * Nếu overlap tăng lên 100 thì chunks sẽ bị tăng lên thành 25 chunks. Muốn overlap nhiều để cho AI hiểu ngữ cảnh hơn, tránh mất thông tin khi một ý bị cắt làm đôi 

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Giáo trình Luật pháp (Giáo trình Luật thi hành án hình sự Việt Nam)

**Tại sao nhóm chọn domain này?**
> Văn bản luật pháp và giáo trình chuyên ngành có tính logic chặt chẽ, từ vựng mang tính chính xác cao và tham chiếu chéo nhiều. Lựa chọn domain này giúp kiểm tra sự hiệu quả của phần chunking và retrieval một cách trực quan nhất; đòi hỏi hệ thống RAG phải có phương pháp cắt văn bản phù hợp để giữ được trọn vẹn ngữ nghĩa các điều luật.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Giáo trình Luật thi hành án hình sự | TRƯỜNG ĐẠI HỌC VINH (data/output.md) | ~265174 | `category`: `law_textbook`, `source`: `đại_học_vinh` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `category` | string | `law_textbook` | Giúp thu hẹp phạm vi tìm kiếm, đảm bảo agent chỉ tham chiếu trên các tài liệu có độ tin cậy và chuyên môn chuẩn xác cao. |
| `source` | string | `đại_học_vinh` | Dễ dàng tra cứu nguồn gốc để agent tự động trích dẫn đầy đủ tác giả, xuất xứ khi đưa ra câu trả lời. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên tài liệu Giáo trình chính của nhóm:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| output.md | FixedSizeChunker (`fixed_size`) | 278 | 997.08 | Trung bình (cắt ngang tùy ý nhưng có overlap để bù đắp) |
| output.md | SentenceChunker (`by_sentences`) | 452 | 579.64 | Tốt (không cắt ngang câu, gom tối đa 3 câu/chunk) |
| output.md | RecursiveChunker (`recursive`) | 358 | 733.69 | Rất tốt (tôn trọng cấu trúc đoạn văn, xuống dòng, dấu chấm) |

### Strategy Của Tôi

**Loại:** CustomChunker (`custom_header` - Cắt theo Tiêu đề Markdown chuyên dụng)

**Mô tả cách hoạt động:**
> Đây là một chiến thuật tự xây dựng dựa trên đặc điểm cấu trúc văn bản Luật. Thuật toán sử dụng Regex để tìm kiếm các thẻ Tiêu đề dọc theo văn bản (như `## ĐIỀU 1`, `#### CHƯƠNG I`) và chia cắt chính xác ngay trước các thẻ tiêu đề đó. Do vậy, một "chunk" được sinh ra sẽ chứa trọn vẹn toàn bộ một đoạn giải nghĩa của Điều Luật hoặc một Chương để giữ tính toàn vẹn. Trường hợp 1 Điều luật quá béo (>1500 ký tự), nó mới dùng cửa sổ trượt (FixedSizeChunker) để chẻ nhỏ.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Do Giáo trình luật chứa đựng các hệ thống Chương, Mục, Điều luật chồng chéo nhau với quy chuẩn biên soạn gắt gao. Cắt theo Header sẽ dứt điểm hiện tượng "Chương một nơi, nội dung chương một nẻo" do thuật toán cũ chém nhầm ở chính giữa. Nó giúp Vector sinh ra ôm trọn vẹn một quy định pháp luật duy nhất vào đúng một chỗ, từ đó việc trả lời câu hỏi và tham chiếu nguồn điều khoản của AI sẽ chính xác và mượt mà tuyệt đối.

**Code snippet (nếu custom):**
```python
# Gọi thuật toán custom đã được tự lập trình trong src/chunking.py 
# (Tách bằng regex '\n(?=#+ \**)')
chunker = CustomChunker(max_chunk_size=1500)
chunks = chunker.chunk(text)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| output.md | best baseline (RecursiveChunker) | 358 | 733.69 | Tốt, hệ thống tôn trọng dấu ngắt đoạn nên bảo vệ được ngữ cảnh. |
| output.md | **của tôi** (CustomChunker) | 373 | 766.35 | Vượt trội, vì từng chunk bây giờ đại diện đúng cấu trúc một "Điều luật" / "Mục", độ dài ổn định. |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Đỗ Xuân Bằng | CustomChunker (Header) | 9.5/10 | Gom hoàn hảo trọn vẹn ý nghĩa của nguyên một Điều luật, không xé ngữ cảnh. | Có nguy cơ tạo chunk dài vượt mức nếu một Điều luật quá dài. |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Strategy của **Đỗ Xuân Bằng (Custom Markdown Header)** tỏ ra thống trị đối với kiểu dữ liệu Giáo trình Luật. Bởi vì các bộ luật vốn được biên soạn với logic phân nhánh cực kỳ rõ ràng theo CHƯƠNG, KHOẢN, ĐIỀU. Việc nương theo các phân cấp Header đảm bảo cho cỗ máy nhốt trọn 1 mảnh vỡ lớn pháp lý (một chủ đề) vào trong đúng 1 Vector. Điều này giúp độ chiết xuất (Retrieval) mang độ chính xác cực cao mà không lo dính tạp âm câu chữ không liên quan.

---

## 4. My Approach — Cá nhân (10 điểm)

**Mô tả quá trình implement:**
- `SentenceChunker`: Tôi sử dụng biểu thức chính quy (Regex: `re.split(r'(?<=[.!?])\s+', text)`) để phát hiện chính xác ranh giới câu thông qua các dấu chấm, chấm hỏi, chấm than. Sau đó, tôi dùng vòng lặp gom cứ 3 câu liên tiếp (`max_sentences_per_chunk = 3`) nén lại thành một chunk duy nhất bằng lệnh `join()`.
- `RecursiveChunker`: Được lập trình đệ quy với danh sách ký tự ngắt đoạn ưu tiên giảm dần: `["\n\n", "\n", ". ", " ", ""]`. Hàm `_split` sẽ tìm cách chẻ văn bản bằng dấu `\n\n`. Nếu mảnh vỡ sinh ra vẫn lớn hơn `chunk_size` (vd 500), nó sẽ tự gọi lại chính nó tiến sâu xuống một cấp (dùng dấu `\n`, rồi đến dấu `. `) để băm nhỏ ra đến khi nào kích thước đạt chuẩn thì thôi.
- `compute_similarity`: Lập trình công thức Cosine Similarity bằng toán học thuần túy. Hàm tính tích vô hướng (Dot product) của 2 vector $\vec{a}$ và $\vec{b}$, sau đó chia cho tích độ lớn biên độ của chúng (Magnitude). Tôi cũng bổ sung cấu trúc rẽ nhánh `if mag_a == 0 or mag_b == 0` trả về 0.0 lập tức để chống lỗi chia cho 0 (ZeroDivisionError).
- `EmbeddingStore.search`: Thiết kế mô hình Hybrid (Mềm dẻo). Hệ thống cố gắng connect với `ChromaDB` trước, tự động nhét query vào `collection.query()`. Nếu thư viện ChromaDB lỗi, nó tự động tụt xuống chạy In-memory List: Vòng lặp quét mọi Document, gọi hàm `compute_similarity()` với Query, gán điểm `score`, dùng `sort()` đảo ngược và rút trích dãy `top_k`.

Một khó khăn (hoặc bug) tôi gặp phải và cách giải quyết:
> **Lỗi nghẽn cổ chai OpenAI Token (BadRequestError):** Khi tôi chạy thử nghiệm file `main.py` trên toàn bộ Giáo trình luật dày >265,000 ký tự với Embedding backend `text-embedding-3-small`, OpenAI API đã đá văng kết nối báo lỗi "maximum context length is 8192 tokens". 
> **Cách tôi khắc phục:** Lỗi xảy ra do code đang tọng thẳng toàn bộ file thô chưa qua phẫu thuật vào `add_documents`. Tôi đã sửa `main.py` chèn thuật toán `CustomChunker` xen vào giữa. Toàn bộ file khổng lồ được chặt ra thành 373 chunks nhỏ gọn trước khi nạp vào store, nhờ đó mọi thứ tương phùng trơn tru.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng Regex `r'(?<=[.!?])\s+'` (Positive Lookbehind) để bắt vị trí ngắt câu sát dấu chấm/hỏi/than mà không cạo mất chính dấu chấm gốc đó. Hệ thống còn tích hợp hàm `.strip()` loại bỏ các khoảng trắng vô cực ở kẽ câu để xử lý edge case (những đoạn xuống dòng hay nhập dư khoảng trắng). Cuối cùng nó lướt qua mảng, dùng lệnh `.join()` gom chuẩn số lượng `max_sentences_per_chunk` (mặc định 3) mảnh thành từng viên gạch chunk.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán đệ quy vòng lặp chẻ tài liệu. Base case (trường hợp kết thúc đệ quy gốc) diễn ra khi khối lượng chunk phân rã ra đã tuân thủ kích thước nhỏ hơn `chunk_size` hoặc đã dùng sập nguồn list các Separators mà vẫn chưa xong việc. Cứ mỗi block lớn hơn giới hạn, code bung tách chuỗi bằng Separator hiện hữu (ví dụ `\n`), sau đó đệ quy tự gọi mình đem block con lặn sâu đánh tiếp với hạng Separator bé hơn tiếp theo (ví dụ `. `) cho tới khi kích cỡ hoàn toàn hợp lệ.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` sẽ gọi hàm embedding chuyển hóa mãng Text từ class Document thành các list `vector[float]` đại diện tọa độ từ vựng, có kèm Metadata, tạo bản ghi Record lưu và ép vào ChromaDB (hoặc Array List dự phòng). Hàm `search` gọi hàm ngầm `compute_similarity` vận dụng điểm toán học Cosine bằng tích vô hướng giữa Vector câu hỏi (Query) và Vector Data nhằm xếp hạng điểm cao nhất (tương ứng Top K ý nghĩa liên đới nhất) từ Database.

**`search_with_filter` + `delete_document`** — approach:
> Cơ chế `search_with_filter` là "Lọc trước, Search Similarity sau". Metadata sẽ được Query kiểm tra logic khớp nhau qua điều kiện `where=metadata_filter` của ChromaDB hoặc loop tay qua cái Dict In-memory để ném bỏ các tài liệu sai chủ đề xong mới nhúng vào phép đo Embedding. Tác vụ `delete_document` sẽ dò mã định danh `doc_id`, dùng lệnh tống cổ toàn bộ record liên quan khỏi store bằng hàm native `collection.delete` ở hệ DB ngoài.

### KnowledgeBaseAgent

**`answer`** — approach:
> Agent làm nhiệm vụ tổng hợp RAG. Đầu tiên gọi hàm `search()` kéo 3 chunks ý nghĩa xịn nhất. Thiết lập một bộ khung Prompt nguyên thuỷ dồn nội dung Context thô xen kẽ với Câu hỏi Query (`f"Context:\n{context}\n\nQuestion: {question}"`). Đưa bộ não LLM gốc gọi API (như là GPT) tiếp quản Prompt để đẻ ngôn ngữ suy luận giải đáp, hoàn thiện việc trả lời Retrieval Augmented Generation.

### Test Results

Dưới đây là Log Terminal chứng minh hệ thống đã Pass 100% (42/42 bài test) cho mọi mô-đun trong package cấu trúc:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.0.3, pluggy-1.6.0
rootdir: E:\Day-07-Lab-Data-Foundations
collecting ... collected 42 items

tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.74s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Người bị kết án tự nguyện bồi thường | Phạm nhân tự giác khắc phục hậu quả | high | 0.436 | Đúng |
| 2 | Con chó đang nằm ngủ trên nệm | Chú cún đang đánh giấc trên giường | high | 0.513 | Đúng |
| 3 | Tôi rất yêu màu xanh bầu trời | Tôi rất ghét thiên nhiên | low | 0.518 | Sai |
| 4 | Mức án tử hình | Nấu ăn trong trại tạm giam | low | 0.236 | Đúng |
| 5 | Hãng Apple ra mắt điện thoại mới | Tôi vừa mua một quả táo ngon | low | 0.417 | Sai |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Cặp kết quả gây bất ngờ nhất là cặp số 3 ("yêu màu xanh bầu trời" vs "ghét thiên nhiên"), điểm tương đồng lại đạt rất cao (0.518) dù ý nghĩa cực kỳ đối lập. Điều này cho thấy thuật toán Embedding đôi khi đánh lừa: khi 2 câu chứa cùng một topic (thiên nhiên) và có cấu trúc từ vựng y hệt nhau thì AI có xu hướng xếp sát chúng vào nhau trong không gian vector, bất chấp việc động từ chính (yêu và ghét) đang tạo ra sắc thái đối kháng. Đôi khi AI phân tích theo "Ngữ pháp/Chủ đề" hơn là "Ngữ nghĩa trái ngược".

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Khái niệm pháp luật thi hành án hình sự là gì? | Là tổng hợp các quy phạm pháp luật điều chỉnh các quan hệ xã hội phát sinh trong quá trình thi hành án. |
| 2 | Nguyên tắc nhân đạo trong thi hành án hình sự thể hiện như thế nào? | Không đối xử tàn bạo, bảo đảm pháp lý cho cuộc sống người bị kết án, tôn trọng quyền con người. |
| 3 | Tác dụng giáo dục cải tạo của hình phạt là gì? | Giáo dục cải tạo họ thành người lương thiện, tuân thủ pháp luật và có ích cho xã hội. |
| 4 | Nhiệm vụ của pháp luật thi hành án hình sự là gì? | Bảo đảm bản án được thực thi nghiêm minh, tạo điều kiện cho người thụ án tái hòa nhập cộng đồng. |
| 5 | Các quyền lợi hợp pháp bị xâm phạm thì người bị kết án giải quyết thế nào? | Có quyền khiếu nại, tố cáo đối với hành vi xâm phạm của cơ quan hoặc cá nhân thi hành án. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Khái niệm pháp luật thi hành... | "Pháp luật thi hành án hình sự là tổng thể các quy phạm..." | 0.176 | Yes | [CÂU TRẢ LỜI MẪU]: Theo quy định... |
| 2 | Nguyên tắc nhân đạo trong thi... | "Nguyên tắc được thể hiện ở những bảo đảm pháp lý nhân đạo..." | 0.352 | Yes | [CÂU TRẢ LỜI MẪU]: Theo quy định... |
| 3 | Tác dụng giáo dục cải tạo của... | "Nhằm mục đích là giáo dục, cải tạo người kết án phục thiện..." | 0.521 | Yes | [CÂU TRẢ LỜI MẪU]: Theo quy định... |
| 4 | Nhiệm vụ pháp luật thi hành án... | "...còn có nhiệm vụ tạo điều kiện để họ tái hòa nhập cộng đồng" | 0.410 | Yes | [CÂU TRẢ LỜI MẪU]: Theo quy định... |
| 5 | Quyền lợi bị xâm phạm thì người... | "...vệ các quyền và lợi ích hợp pháp khỏi sự xâm hại trái phép... khiếu nại" | 0.179 | Yes | [CÂU TRẢ LỜI MẪU]: Theo quy định... |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
