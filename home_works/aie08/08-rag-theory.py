### RAG là gì? và tại sao cần RAG?

# vấn đề: LLM không biết dữ liệu của BẠN
# chỉ biết những gì đã học được từ dữ liệu huấn luyện, không biết dữ liệu riêng, mới
# => dễ bịa khi thiếu thông tin (hallucination)
# example: 
# 1) Why hotel are expensive in the weekend? => LLM: Because of high demand in the weekend.
# 2) Why Muong Thanh hotal are very expensive this weekend? => LLM không biết => bịa liền.


## 1. RAG: Retrieval Augmented Generation
# tra (retrieval) cứu dữ liệu riêng của bạn -> đưa vào prompt cho LLM (augment) -> model trả lời (generation) dựa trên tài liệu đó.

# RAG giải quyết vấn đề gì?
# => Dùng dữ liệu riêng của bạn (nội bộ, db, tri thức sản phẩm mới) - thứ model chưa từng thấy khi huấn luyện
# => cập nhật dữ liệu mới (new knowledge) - dễ dàng, không cần huấn luyện lại model (tốn kém)
# => giảm bịa (hallucination) - model trả lời dựa trên dữ liệu riêng của bạn, không bịa.
# => trích dẫn được nguồn (source) - biết được câu trả lời đến từ tài liệu/ đoạn nào


## 2. sơ đồ tổng quan 2 pha của RAG:

# 1.indexing: làm 1 lần (offline) - đưa dữ liệu riêng của bạn vào vector database (vector embedding)
# - tài liệu (text, pdf, web,...) -> chunking -> embedding -> vector database

# 2. query: làm mỗi lần có câu hỏi (online) - tìm kiếm dữ liệu riêng của bạn trong vector database -> đưa vào prompt cho LLM -> model trả lời dựa trên tài liệu đó.
# câu hỏi -> embedding -> vector database -> tìm các đoạn (chunk) tương tự nhất -> đưa vào prompt cho LLM -> model trả lời dựa trên tài liệu đó.


### Xây dựng RAG pipeline từ A-Z

# Ta sẽ xây pipeline RAG hoàn chỉnh theo các bước:
# Bước 1 - Cấu hình: API key cho LLM/embedding
# Bước 2 - Dữ liệu (Document): chunk tài liệu tự nhiên, đủ nghĩa
# Bước 3 - Embedding: biến text thành vector số
# Bước 4 - Indexing: nhúng tài liệu vào không gian vector
# Bước 5 - Retrieval: truy vấn → top-k chunk gần nhất (cosine similarity)
# Bước 6 - Augment: ghép chunk tìm được vào prompt
# Bước 7 - Generation: gửi prompt tới LLM → nhận câu trả lời
# Bước 8 - Pipeline hoàn chỉnh: gom các bước trên, nếu không tìm thấy → fallback


# ---------------------------------------------------------------------------
# Bước 1: Cấu hình - Settings + Data Models
# ---------------------------------------------------------------------------

# Ý tưởng: RAG cần 2 thành phần dữ liệu chính:
# - DocumentChunk: 1 đoạn tài liệu gốc + embedding vector của nó
# - RAGResponse: câu trả lời + danh sách nguồn tham khảo
# Settings đọc GEMINI_API_KEY từ .env để gọi API

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GEMINI_API_KEY: str | None = None
    model_config = {"env_file": ".env", "extra": "ignore"}


class DocumentChunk(BaseModel):
    """1 chunk tài liệu: id, title, text, embedding (vector số)."""
    id: int = Field(..., description="ID định danh duy nhất của chunk")
    title: str = Field(..., description="Tiêu đề/Nhãn của chunk tài liệu")
    text: str = Field(..., description="Nội dung chi tiết của chunk tài liệu")
    embedding: list[float] | None = Field(
        default=None, description="Vector embedding của chunk"
    )


class RAGResponse(BaseModel):
    """Kết quả pipeline RAG: answer + danh sách nguồn + chế độ."""
    answer: str = Field(..., description="Câu trả lời cuối cùng từ hệ thống")
    retrieved_sources: list[tuple[int, str, float]] = Field(
        ...,
        description="Danh sách nguồn được dùng (ID, Title, Cosine Similarity Score)",
    )
    mode: str = Field(..., description="Chế độ chạy (Real API hoặc Mock Simulation)")


# ---------------------------------------------------------------------------
# Bước 2: Dữ liệu mẫu (Document Chunks) - 10 chunk tự nhiên, đủ nghĩa
# ---------------------------------------------------------------------------

# Mỗi chunk là 1 đoạn hoàn chỉnh về Sunhouse Smart Cooker.
# Chunk phải tự đủ nghĩa, không phụ thuộc chunk khác → mỗi chunk là 1 đơn vị tri thức.
# Khi LLM hỏi "Làm sao kết nối Wifi?", ta tìm chunk có embedding gần nhất.
# ta đang cố tình để thiếu list vector embedding, để bước 4 (indexing) sẽ nhúng vector embedding cho từng chunk.

CHUNKS_DATA = [
    DocumentChunk(
        id=1,
        title="Giới thiệu chung về Sunhouse Smart Cooker",
        text="Thiết bị Sunhouse Smart Cooker là nồi đa năng thông minh tích hợp hơn 15 chế độ nấu tự động, màn hình cảm ứng OLED sắc nét và hỗ trợ kết nối Wifi băng tần 2.4GHz để điều khiển từ xa.",
    ),
    DocumentChunk(
        id=2,
        title="Hướng dẫn kết nối Wifi cho Sunhouse Smart Cooker",
        text="Để kết nối Sunhouse Smart Cooker với mạng Wifi, trước hết hãy nhấn giữ nút Wifi trên bảng điều khiển trong 5 giây cho đến khi đèn báo nháy nhanh. Tiếp theo, mở ứng dụng SmartLife trên điện thoại, chọn Thêm thiết bị và làm theo hướng dẫn kết nối trên ứng dụng.",
    ),
    DocumentChunk(
        id=3,
        title="Chế độ nấu áp suất an toàn",
        text="Khi sử dụng chế độ nấu áp suất của Sunhouse Smart Cooker, van xả áp phải luôn ở vị trí đóng (Sealing). Tuyệt đối không cố gắng mở nắp nồi khi cột chỉ thị áp suất màu đỏ vẫn đang nổi lên. Hãy đợi nồi tự hạ áp suất hoặc nhấn nút xả áp thủ công trước khi mở.",
    ),
    DocumentChunk(
        id=4,
        title="Vệ sinh lòng nồi và khay nước ngưng tụ",
        text="Lòng nồi của Sunhouse Smart Cooker được phủ lớp chống dính gốm cao cấp. Hãy vệ sinh lòng nồi bằng nước ấm, xà phòng dịu nhẹ và bọt biển mềm. Không dùng búi sắt hoặc chất tẩy rửa mạnh. Khay chứa nước ngưng tụ ở mặt sau cần tháo và đổ nước sau mỗi lần nấu.",
    ),
    DocumentChunk(
        id=5,
        title="Chính sách bảo hành chính hãng",
        text="Thiết bị Sunhouse Smart Cooker được bảo hành chính hãng 24 tháng đối với các lỗi phần cứng phát sinh từ phía nhà sản xuất (như hỏng bảng điều khiển, lỗi cảm biến nhiệt). Các phụ kiện đi kèm như muỗng, xửng hấp được bảo hành 12 tháng.",
    ),
    DocumentChunk(
        id=6,
        title="Chính sách đổi trả sản phẩm",
        text="Khách hàng được quyền đổi mới sản phẩm Sunhouse Smart Cooker miễn phí trong vòng 7 ngày đầu kể từ ngày mua nếu sản phẩm gặp lỗi phần cứng kỹ thuật được xác nhận bởi trung tâm bảo hành. Sản phẩm đổi trả phải đầy đủ hộp và phụ kiện đi kèm.",
    ),
    DocumentChunk(
        id=7,
        title="Mã lỗi E1 và cách khắc phục",
        text="Lỗi E1 hiển thị trên màn hình Sunhouse Smart Cooker cảnh báo tình trạng nồi bị quá nhiệt (nhiệt độ lòng nồi vượt mức 200 độ C do thiếu nước hoặc bị cháy khét đáy). Cách xử lý: Rút phích cắm điện ngay lập tức, để nồi nguội hoàn toàn trong ít nhất 15 phút, thêm nước trước khi nấu tiếp.",
    ),
    DocumentChunk(
        id=8,
        title="Chế độ nấu chậm (Slow Cook)",
        text="Chế độ Slow Cook của Sunhouse Smart Cooker duy trì nhiệt độ ổn định ở mức 85-90 độ C trong thời gian dài (từ 2 đến 8 giờ tùy cài đặt). Chế độ này lý tưởng cho các món hầm xương, kho cá giúp giữ trọn vẹn hương vị và dưỡng chất.",
    ),
    DocumentChunk(
        id=9,
        title="Tải thêm công thức nấu ăn mới",
        text="Bạn có thể tải thêm hàng trăm công thức nấu ăn miễn phí thông qua kho công thức trực tuyến trên ứng dụng SmartLife. Các công thức mới được cập nhật tự động định kỳ vào ngày 1 hàng tháng.",
    ),
    DocumentChunk(
        id=10,
        title="Thông tin liên hệ hỗ trợ kỹ thuật",
        text="Mọi thắc mắc kỹ thuật về Sunhouse Smart Cooker xin vui lòng liên hệ tổng đài chăm sóc khách hàng 1900-8198 (hoạt động từ 8h00 đến 21h00 tất cả các ngày trong tuần) hoặc gửi email trực tiếp tới support@sunhouse.vn.",
    ),
]


# ---------------------------------------------------------------------------
# Bước 3: Embedding - biến text thành vector số (dùng Gemini API)
# ---------------------------------------------------------------------------

# Embedding là quá trình biến câu văn bản thành 1 vector số (list[float]).
# Văn bản giống nhau → vector gần nhau về mặt hình học.
# Ta dùng model "gemini-embedding-001" của Google để nhúng văn bản.
# Nếu API lỗi, trả về mảng rỗng và mode = "error".

import numpy as np
from google import genai
from google.genai import types


settings = Settings()


def get_embedding(texts: list[str]) -> tuple[list[list[float]], str]:
    """Nhúng 1 danh sách text thành vector số. Trả về (embeddings, mode)."""
    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        gemini_response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )
        embeddings = [emb.values for emb in gemini_response.embeddings]
        return embeddings, "gemini-embedding-001 api"
    except Exception as e:
        print(f"Lỗi khi gọi API embedding: {e}")
        return [], "error"


# ---------------------------------------------------------------------------
# Cosine Similarity - đo độ tương đồng giữa 2 vector
# ---------------------------------------------------------------------------

# Cosine similarity đo góc giữa 2 vector: cos(theta) = (A·B) / (|A| * |B|)
# Giá trị từ -1 (ngược chiều) đến 1 (cùng chiều).
# Với embedding văn bản, giá trị càng gần 1 càng tương đồng về ngữ nghĩa.
# Nếu 1 trong 2 vector bằng 0, trả về 0.0.


def compute_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Tính điểm tương đồng cosine giữa 2 vector."""
    np_a = np.array(vec_a)
    np_b = np.array(vec_b)
    dot_product = np.dot(np_a, np_b)
    norm_a = np.linalg.norm(np_a)
    norm_b = np.linalg.norm(np_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Bước 4: Indexing - nhúng tất cả chunk vào vector space (chạy 1 lần)
# ---------------------------------------------------------------------------

# Đây là bước offline: với mỗi chunk trong CHUNKS_DATA, gọi get_embedding()
# để lấy vector, rồi gắn vào chunk.embedding.
# Kết quả: mỗi chunk có 1 vector số để sau này so sánh với câu hỏi.

# Code thực tế ở dưới hàm main():
#   chunk_texts = [chunk.text for chunk in CHUNKS_DATA]
#   embeddings, source = get_embedding(chunk_texts)
#   for i, emb in enumerate(embeddings):
#       CHUNKS_DATA[i].embedding = emb


# ---------------------------------------------------------------------------
# Bước 5: Retrieval - tìm top-k chunk gần nhất với câu hỏi
# ---------------------------------------------------------------------------

# Khi có câu hỏi mới, ta nhúng câu hỏi → vector q.
# So sánh q với embedding của từng chunk bằng cosine similarity.
# Sắp xếp giảm dần theo điểm số, lấy top_k chunk gần nhất.
# Hàm trả về list[(chunk_id, title, score)].


def retrieve_relevant_chunks(query: str, top_k: int = 3) -> list[tuple[int, str, float]]:
    """Truy xuất các chunk tài liệu liên quan đến truy vấn dựa trên embedding."""
    emb_query, _ = get_embedding([query])
    if not emb_query:
        print(f"Lỗi khi tạo embedding cho truy vấn: {query}")
        return []

    query_vector = emb_query[0]
    similarities = [
        (chunk.id, chunk.title, compute_cosine_similarity(query_vector, chunk.embedding))
        for chunk in CHUNKS_DATA if chunk.embedding is not None
    ]
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:top_k]


# ---------------------------------------------------------------------------
# Bước 6: Augment - ghép chunk tìm được vào prompt cho LLM
# ---------------------------------------------------------------------------

# Lấy top-k chunk tìm được ở bước 5, ghép thành 1 ngữ cảnh duy nhất.
# Dùng SYSTEM_PROMPT để định hướng LLM: chỉ trả lời dựa trên ngữ cảnh,
# không được bịa, không suy đoán. Nếu không đủ thông tin → nói "không có".
# Kết quả là 1 prompt hoàn chỉnh để gửi tới LLM.

SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia hỗ trợ kỹ thuật và chăm sóc khách hàng của sản phẩm Sunhouse Smart Cooker. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng bằng cách DỰA TRÊN NGỮ CẢNH được cung cấp từ tài liệu. Hãy tuân thủ nghiêm ngặt các yêu cầu sau:
1. Chỉ trả lời dựa trên thông tin trong đoạn trích. không suy đoán hoặc thêm thông tin bên ngoài.
2. Nếu đoạn trích không đủ thông tin để trả lời, hãy nói "Tài liệu không đề cập đến vấn đề này."
3. Trả lời theo nội dung trích dẫn, cùng với toàn bộ đoạn trích (nếu có) chứa câu trả lời.
4. Trả lời bằng tiếng Việt."""


def build_rag_prompt(query: str, results: list[dict]) -> str:
    """Ghép SYSTEM_PROMPT + ngữ cảnh + câu hỏi thành prompt hoàn chỉnh."""
    context = "\n\n---\n\n".join(
        [f"[Chunk {r['index']}] {r['text']}" for r in results]
    )
    return f"""{SYSTEM_PROMPT}
---NGỮ CẢNH---
{context}
---KẾT THÚC NGỮ CẢNH---

Câu hỏi: {query}

Trả lời:"""


# ---------------------------------------------------------------------------
# Bước 7: Generation - gọi LLM để sinh câu trả lời
# ---------------------------------------------------------------------------

# Gửi prompt (đã ghép từ bước 6) đến Gemini API.
# Model "gemini-3.1-flash-lite" được chọn vì nhanh, rẻ.
# temperature=0.1 để đầu ra ít ngẫu nhiên, bám sát ngữ cảnh.
# Trả về text response từ LLM.


def ask_llm(prompt: str) -> str:
    """Gửi prompt tới Gemini API, trả về text response."""
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    resp = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1),
    )
    return resp.text


# ---------------------------------------------------------------------------
# Bước 8: Pipeline hoàn chỉnh - gom tất cả các bước
# ---------------------------------------------------------------------------

# Pipeline đầy đủ:
# 1. Nhúng toàn bộ CHUNKS_DATA (indexing) - chạy 1 lần
# 2. Với mỗi câu hỏi:
#    a. Nhúng câu hỏi
#    b. Tính cosine similarity với tất cả chunk
#    c. Lấy top-k chunk
#    d. Kiểm tra ngưỡng similarity (0.7), nếu thấp hơn → fallback
#    e. Ghép context → prompt → gửi LLM
#    f. In kết quả + nguồn tham khảo
# 3. Fallback: nếu không tìm thấy chunk phù hợp, trả lời mặc định


def main():
    if not settings.GEMINI_API_KEY:
        print("Lỗi: GEMINI_API_KEY chưa được cấu hình trong .env")
        return

    # --- INDEXING: nhúng tất cả chunk ---
    chunk_texts = [chunk.text for chunk in CHUNKS_DATA]
    embeddings, source = get_embedding(chunk_texts)

    if not embeddings:
        print("Lỗi: Không thể tạo embedding cho các chunk. Kiểm tra API key và kết nối.")
        return

    for i, emb in enumerate(embeddings):
        CHUNKS_DATA[i].embedding = emb

    print(f"Đã tạo embedding cho {len(CHUNKS_DATA)} chunk từ nguồn: {source}")

    # --- QUERY: test với 3 câu hỏi ---
    test_queries = [
        "Làm thế nào để kết nối nồi Sunhouse Smart Cooker với Wifi?",
        "Chế độ nấu áp suất có an toàn không?",
        "Vì sao thời tiết Đà Nẵng hôm nay lại mưa to?",
    ]

    SIMILARITY_THRESHOLD = 0.7

    for query in test_queries:
        relevant_chunks = retrieve_relevant_chunks(query, top_k=4)

        # Fallback: không tìm thấy chunk phù hợp
        if not relevant_chunks or relevant_chunks[0][2] < SIMILARITY_THRESHOLD:
            print(f"\nCâu hỏi: {query}")
            print("Câu trả lời: Không tìm thấy thông tin liên quan trong tài liệu.")
            print("Nguồn tham khảo: (không có)")
            continue

        # Augment: ghép chunk vào prompt
        rag_prompt = build_rag_prompt(query, [
            {"index": chunk_id, "text": next(chunk.text for chunk in CHUNKS_DATA if chunk.id == chunk_id)}
            for chunk_id, _, _ in relevant_chunks
        ])

        # Generation: gọi LLM
        answer = ask_llm(rag_prompt)

        print(f"\nCâu hỏi: {query}")
        print(f"Câu trả lời: {answer}")
        print("Nguồn tham khảo:")
        for chunk_id, title, score in relevant_chunks:
            print(f"  - Chunk ID: {chunk_id}, Title: {title}, Cosine Similarity: {score:.4f}")


# ---------------------------------------------------------------------------
# Chạy pipeline
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     main()

# Kết quả mong đợi:
#   Câu 1 "kết nối Wifi" → tìm đúng chunk #2, LLM trả lời dựa trên hướng dẫn.
#   Câu 2 "áp suất an toàn" → tìm đúng chunk #3, LLM trích dẫn quy tắc an toàn.
#   Câu 3 "thời tiết Đà Nẵng" → similarity thấp → fallback "Không tìm thấy".
#   + Mỗi câu in kèm danh sách nguồn (ID, Title, Score) để kiểm tra độ tin cậy.

