import io
import sys
import numpy as np
from google import genai
from pydantic_settings import BaseSettings
from google.genai import types
from pydantic import BaseModel, Field

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")




class Settings(BaseSettings):
    GEMINI_API_KEY: str | None = None
    model_config = {"env_file": ".env", "extra": "ignore"}


class DocumentChunk(BaseModel):
    """Lớp dữ liệu đại diện cho một chunk tài liệu."""

    id: int = Field(..., description="ID định danh duy nhất của chunk")
    title: str = Field(..., description="Tiêu đề/Nhãn của chunk tài liệu")
    text: str = Field(..., description="Nội dung chi tiết của chunk tài liệu")
    embedding: list[float] | None = Field(
        default=None, description="Vector embedding của chunk"
    )


class RAGResponse(BaseModel):
    """Lớp dữ liệu chứa kết quả phản hồi của hệ thống RAG."""

    answer: str = Field(..., description="Câu trả lời cuối cùng từ hệ thống")
    retrieved_sources: list[tuple[int, str, float]] = Field(
        ...,
        description="Danh sách nguồn được dùng (ID, Title, Cosine Similarity Score)",
    )
    mode: str = Field(..., description="Chế độ chạy (Real API hoặc Mock Simulation)")


# ---------------------------------------------------------------------------
# 2. REAL DOCUMENT CHUNKS (Yêu cầu 1: 8-15 chunk tự đủ nghĩa)
# ---------------------------------------------------------------------------
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


settings = Settings()

SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia hỗ trợ kỹ thuật và chăm sóc khách hàng của sản phẩm Sunhouse Smart Cooker. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng bằng cách DỰA TRÊN NGỮ CẢNH được cung cấp từ tài liệu. Hãy tuân thủ nghiêm ngặt các yêu cầu sau:
1. Chỉ trả lời dựa trên thông tin trong đoạn trích. không suy đoán hoặc thêm thông tin bên ngoài.
2. Nếu đoạn trích không đủ thông tin để trả lời, hãy nói "Tài liệu không đề cập đến vấn đề này."
3. Trả lời theo nội dung trích dẫn, cùng với toàn bộ đoạn trích (nếu có) chứa câu trả lời.
4. Trả lời bằng tiếng Việt."""


def ask_llm(prompt: str) -> str:
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    resp = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
        temperature=0.1
        ),
        
    )
    return resp.text


def build_rag_prompt(query: str, results: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        [f"[Chunk {r['index']}] {r['text']}" for r in results]
    )
    return f"""{SYSTEM_PROMPT}
---NGỮ CẢNH---
{context}
---KẾT THÚC NGỮ CẢNH---

Câu hỏi: {query}

Trả lời:"""


def get_embedding(texts: list[str]) -> tuple[list[list[float]], str]:
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
    # Sắp xếp theo điểm tương đồng giảm dần và lấy top_k
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:top_k]


def main():
    if not settings.GEMINI_API_KEY:
        print("Lỗi: GEMINI_API_KEY chưa được cấu hình trong .env")
        return
    
    chunk_texts = [chunk.text for chunk in CHUNKS_DATA]
    embeddings, source = get_embedding(chunk_texts)

    if not embeddings:
        print("Lỗi: Không thể tạo embedding cho các chunk. Kiểm tra API key và kết nối.")
        return

    for i, emb in enumerate(embeddings):
        CHUNKS_DATA[i].embedding = emb

    print(f"Đã tạo embedding cho {len(CHUNKS_DATA)} chunk từ nguồn: {source}")
    
    test_queries = [
        "Làm thế nào để kết nối nồi Sunhouse Smart Cooker với Wifi?",
        "Chế độ nấu áp suất có an toàn không?",
        "Vì sao thời tiết Đà Nẵng hôm nay lại mưa to?",
    ]
    
    # for chunk in CHUNKS_DATA:
    #     print(f"Chunk ID: {chunk.id}, Title: {chunk.title}, Embedding Length: {len(chunk.embedding) if chunk.embedding else 'None'}")
    
    # Thử nghiệm truy vấn và lấy kết quả RAG
    SIMILARITY_THRESHOLD = 0.7

    for query in test_queries:
        relevant_chunks = retrieve_relevant_chunks(query, top_k=4)

        if not relevant_chunks or relevant_chunks[0][2] < SIMILARITY_THRESHOLD:
            print(f"\nCâu hỏi: {query}")
            print("Câu trả lời: Không tìm thấy thông tin liên quan trong tài liệu.")
            print("Nguồn tham khảo: (không có)")
            continue

        rag_prompt = build_rag_prompt(query, [
            {"index": chunk_id, "text": next(chunk.text for chunk in CHUNKS_DATA if chunk.id == chunk_id)}
            for chunk_id, _, _ in relevant_chunks
        ])
        
        answer = ask_llm(rag_prompt)
        
        print(f"\nCâu hỏi: {query}")
        print(f"Câu trả lời: {answer}")
        print("Nguồn tham khảo:")
        for chunk_id, title, score in relevant_chunks:
            print(f"  - Chunk ID: {chunk_id}, Title: {title}, Cosine Similarity: {score:.4f}")

if __name__ == "__main__":
    main()