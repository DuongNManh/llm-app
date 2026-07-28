import json
import time
from pydantic_settings import BaseSettings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from google import genai


class Settings(BaseSettings):
    GEMINI_API_KEY: str | None = None
    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()

SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia hỗ trợ kỹ thuật và chăm sóc khách hàng của sản phẩm Samsung Smart Phone. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng bằng cách DỰA TRÊN NGỮ CẢNH được cung cấp từ tài liệu. Hãy tuân thủ nghiêm ngặt các yêu cầu sau:
1. Chỉ trả lời dựa trên thông tin trong đoạn trích. không suy đoán hoặc thêm thông tin bên ngoài.
2. Nếu đoạn trích không đủ thông tin để trả lời, hãy nói "Tài liệu không đề cập đến vấn đề này."
3. Trả lời theo nội dung trích dẫn, cùng với toàn bộ đoạn trích (nếu có) chứa câu trả lời.
4. Trả lời bằng tiếng Việt, thân thiện và dễ hiểu, không sử dụng từ ngữ chuyên ngành quá khó hiểu."""


PDF_PATH = "documents/huongdansudungSamSung.pdf"
CHROMA_DIR = "./chroma_db"
EMBED_BATCH_SIZE = 10
EMBED_BATCH_SLEEP = 10
RETRIEVER_K = 4
HYBRID_K = 15

# ─────────────── TEST CASES ───────────────

TEST_CASES = [
    {"id": 1, "question": "Hãy hướng dẫn kết nối Wifi cho điện thoại Samsung.",
     "expected_answer_contains": ["Cài đặt", "Kết nối", "Wi-Fi", "mạng Wi-Fi"], "expected_source_pages": [65, 66], "category": "general"},
    {"id": 2, "question": "Cách chụp màn hình trên điện thoại Samsung?",
     "expected_answer_contains": ["phím Cạnh", "Giảm âm lượng", "cùng lúc"], "expected_source_pages": [30], "category": "general"},
    {"id": 3, "question": "Pin điện thoại Samsung nên sạc như thế nào cho đúng?",
     "expected_answer_contains": ["sạc", "pin", "USB Type-C"], "expected_source_pages": [13, 14], "category": "general"},
    {"id": 4, "question": "Làm thế nào để chuyển dữ liệu từ máy cũ sang máy Samsung mới?",
     "expected_answer_contains": ["Smart Switch", "dữ liệu", "chuyển"], "expected_source_pages": [21, 22], "category": "general"},
    {"id": 5, "question": "Làm sao để vào Internet qua Wifi trên máy Samsung?",
     "expected_answer_contains": ["Cài đặt", "Kết nối", "Wi-Fi"], "expected_source_pages": [65, 66], "category": "semantic_gap"},
    {"id": 6, "question": "Điện thoại Samsung bị treo logo, làm thế nào để khắc phục?",
     "expected_answer_contains": ["nhấn và giữ", "phím Cạnh", "phím Giảm âm lượng", "7 giây", "khởi động lại"], "expected_source_pages": [19], "category": "semantic_gap"},
    {"id": 7, "question": "Máy Samsung của tôi bị đơ, không bấm được gì, phải làm sao?",
     "expected_answer_contains": ["nhấn và giữ", "phím Cạnh", "phím Giảm âm lượng", "khởi động lại"], "expected_source_pages": [19], "category": "semantic_gap"},
    {"id": 8, "question": "Samsung của tôi bị nóng quá, có sao không?",
     "expected_answer_contains": ["nóng", "thiết bị", "sạc", "ứng dụng"], "expected_source_pages": [6, 7, 8], "category": "semantic_gap"},
    {"id": 9, "question": "Làm cách nào sao chép ảnh từ Samsung qua máy vi tính?",
     "expected_answer_contains": ["Smart Switch", "máy tính", "dữ liệu"], "expected_source_pages": [22], "category": "semantic_gap"},
    {"id": 10, "question": "SM-A125F/DS dùng loại thẻ SIM nào?",
     "expected_answer_contains": ["nano SIM", "SIM"], "expected_source_pages": [15, 16], "category": "code_model"},
    {"id": 11, "question": "Điện thoại Samsung có hỗ trợ Dolby Atmos không?",
     "expected_answer_contains": ["Dolby Atmos", "âm thanh", "Cài đặt"], "expected_source_pages": [72], "category": "code_model"},
    {"id": 12, "question": "Smart Switch có thể chuyển dữ liệu bằng cách nào?",
     "expected_answer_contains": ["Smart Switch", "Không dây", "máy tính", "dữ liệu"], "expected_source_pages": [21, 22], "category": "code_model"},
    {"id": 13, "question": "Samsung Members giúp ích gì khi máy gặp vấn đề?",
     "expected_answer_contains": ["Samsung Members", "hỗ trợ", "chẩn đoán"], "expected_source_pages": [56], "category": "code_model"},
    {"id": 14, "question": "Giá bán của sản phẩm này là bao nhiêu?",
     "expected_answer_contains": ["Tài liệu không đề cập", "không đề cập"], "expected_source_pages": [], "category": "out_of_scope"},
    {"id": 15, "question": "Bảo hành điện thoại Samsung bao lâu?",
     "expected_answer_contains": ["Tài liệu không đề cập", "không đề cập"], "expected_source_pages": [], "category": "out_of_scope"},
]


# ─────────────── INDEXING ───────────────

def load_pdf(file_path: str = PDF_PATH) -> list:
    """Load PDF và trả về danh sách các trang dưới dạng Document."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_documents_v1(docs: list, chunk_size: int = 512, chunk_overlap: int = 128) -> list:
    """Split các trang thành các đoạn nhỏ hơn để tạo embeddings.

    Cải tiến so với baseline (1000/200):
    - Chunk_size=512: mỗi chunk tập trung vào 1 ý, precision cao hơn
    - Chunk_overlap=128: đủ để không mất context biên
    - Separators ưu tiên: headers -> paragraphs -> sentences
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=[
            "\n## ", "\n### ", "\n#### ",
            "\n\n",
            "\n",
            ".", "?", "!",
            ";", ",", " ",
            "",
        ],
    )
    return splitter.split_documents(docs)


def create_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Tạo embeddings từ Google Gemini Embeddings API."""
    gemini_api_key = settings.GEMINI_API_KEY
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY chưa được cấu hình trong .env")
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        api_key=gemini_api_key,
    )


def create_vectorstore(
    documents: list,
    persist_dir: str = CHROMA_DIR,
    batch_size: int = EMBED_BATCH_SIZE,
) -> Chroma:
    # if Path(persist_dir).exists():
    #     shutil.rmtree(persist_dir)

    embeddings = create_embeddings()
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    total = len(documents)
    for i in range(0, total, batch_size):
        batch = documents[i : i + batch_size]
        vectorstore.add_documents(batch)
        print(f"  → Batch {i//batch_size + 1}/{(total-1)//batch_size + 1}: {len(batch)} chunks")
        if i + batch_size < total:
            time.sleep(EMBED_BATCH_SLEEP)

    return vectorstore


def load_vectorstore(persist_dir: str = CHROMA_DIR) -> Chroma:
    """hàm load lại vectorstore từ Chroma DB đã lưu, để không phải re-index lại."""
    # biến embeddings được tạo lại để đảm bảo tương thích với Chroma DB đã lưu, vì embeddings có thể thay đổi theo model hoặc version.
    embeddings = create_embeddings()
    return Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


# ─────────────── EVALUATION METRICS ───────────────

def normalize_text(text: str) -> str:
    return text.lower().strip()


def keyword_coverage(answer: str, expected_keywords: list) -> float:
    if not expected_keywords:
        return 1.0
    answer_lower = normalize_text(answer)
    matches = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return matches / len(expected_keywords)


def check_context_recall(retrieved_docs: list, expected_pages: list) -> bool:
    if not expected_pages:
        return True
    retrieved_pages = set()
    for doc in retrieved_docs:
        p = doc.metadata.get("page")
        if p is not None:
            try:
                retrieved_pages.add(int(p) + 1)
            except (ValueError, TypeError):
                retrieved_pages.add(p)
    return any(ep in retrieved_pages for ep in expected_pages)


def save_report(all_results: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    txt_path = path.replace(".json", ".txt")
    lines = []
    for variant_name, result in all_results.items():
        lines.append(f"{'='*60}")
        lines.append(f"VARIANT: {variant_name}")
        lines.append(f"{'='*60}")
        lines.append(f"  Total cases : {result['num_cases']}")
        lines.append(f"  Context Rec : {result['context_recall']}")
        lines.append(f"  Keyword Cov : {result['keyword_coverage']}")
        lines.append(f"  Avg Latency : {result['avg_latency']}s")
        lines.append(f"{'='*60}")
        lines.append("")
        for r in result["details"]:
            status = "✓" if (r["context_recall"] if r["expected_pages"] else True) else "✗"
            lines.append(f"#{r['id']} [{status}] {r['category']:14s}  CR={r['context_recall']}  KW={r['keyword_coverage']:.2f}  {r['latency_total']:.1f}s")
            lines.append(f"     pages -> {r['retrieved_pages']} (expected {r['expected_pages']})")
            lines.append(f"     Q: {r['question'][:80]}")
            lines.append(f"     A: {r['answer_preview'][:200]}")
            lines.append("")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  → Saved {path} and {txt_path}")


def run_evaluation(name: str, retriever, chain, test_cases: list) -> dict:
    results = []
    total_cr = total_latency = 0
    total_kw = 0.0
    cr_count = kw_count = 0

    for tc in test_cases:
        question = tc["question"]
        expected_kw = tc.get("expected_answer_contains", [])
        expected_pages = tc.get("expected_source_pages", [])

        t0 = time.time()
        retrieved_docs = retriever.invoke(question)
        t1 = time.time()
        answer = chain.invoke(question)
        t2 = time.time()

        cr = check_context_recall(retrieved_docs, expected_pages)
        kw = keyword_coverage(answer, expected_kw)

        retrieved_page_list = []
        for d in retrieved_docs:
            p = d.metadata.get("page")
            if p is not None:
                try:
                    retrieved_page_list.append(int(p) + 1)
                except (ValueError, TypeError):
                    retrieved_page_list.append(p)
            else:
                retrieved_page_list.append(None)

        results.append({
            "id": tc["id"], "category": tc["category"], "question": question,
            "answer_preview": answer[:400], "context_recall": cr,
            "keyword_coverage": round(kw, 4), "latency_total": round(t2 - t0, 3),
            "retrieved_pages": retrieved_page_list, "expected_pages": expected_pages,
        })

        if expected_pages:
            total_cr += int(cr); cr_count += 1
        if expected_kw:
            total_kw += kw; kw_count += 1
        total_latency += t2 - t0

        status = "✓" if (cr if expected_pages else True) else "✗"
        print(f"  [{status}] #{tc['id']} {tc['category']:14s} | CR={cr} KW={kw:.2f} | pages={retrieved_page_list} | {t2-t0:.1f}s")

    return {
        "variant": name, "num_cases": len(results),
        "context_recall": round(total_cr / cr_count, 4) if cr_count else None,
        "keyword_coverage": round(total_kw / kw_count, 4) if kw_count else None,
        "avg_latency": round(total_latency / len(results), 3),
        "details": results,
    }


# ─────────────── RETRIEVAL ───────────────

def format_docs(docs: list) -> str:
    """Format các Document thành chuỗi để đưa vào prompt."""
    # Nếu muốn debug, có thể in ra các chunk retrieved để xem chất lượng retrieval.
    print(f"\n>>>> Retrieved {len(docs)} chunks:")
    for i, d in enumerate(docs):
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page", "?")
        print(f"  [{i}] page={page} | source={src} | {d.page_content[:120]}...")
    print()
    return "\n\n---\n\n".join(
        f"[Trang {doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in docs
    )

from langchain_core.vectorstores import VectorStoreRetriever

def build_vector_retriever(vectorstore, k: int = RETRIEVER_K) -> VectorStoreRetriever:
    """Pure vector search retriever (baseline)."""
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_bm25_retriever(documents, k: int = RETRIEVER_K):
    """BM25 retriever (keyword search)."""
    from langchain_community.retrievers import BM25Retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    return bm25_retriever

from langchain_classic.retrievers import EnsembleRetriever

def build_hybrid_retriever(vectorstore, documents, k: int = HYBRID_K) -> EnsembleRetriever:
    """Hybrid retriever: BM25 (keyword) + Vector (semantic).

    Kỹ thuật cải tiến chính:
    - BM25 bắt chính xác từ khóa (model number, tên lỗi)
    - Vector hiểu ngữ nghĩa (synonym, paraphrase)
    - Kết hợp bằng weighted fusion (0.5/0.5)
    - k=15 lấy rộng làm ứng viên
    """
    # nghĩa là như thế nào? ta có vectorstore (Chroma) đã được tạo từ các chunk, và ta có documents gốc (các trang PDF). BM25 sẽ tìm kiếm trên documents gốc, còn vectorstore sẽ tìm kiếm trên embeddings của các chunk. Kết quả của 2 retriever này sẽ được kết hợp lại bằng EnsembleRetriever.
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever

    # từ documents gốc, tạo BM25 retriever, BM25 sẽ tìm kiếm các đoạn có từ khóa liên quan
    bm25_retriever = BM25Retriever.from_documents(documents)
    # set k cho BM25 retriever, nghĩa là lấy top k kết quả từ BM25
    # chẳng lẽ bm25 sẽ duyệt hết documents để tìm top k? đúng vậy, BM25 sẽ tính score cho từng document dựa trên từ khóa và lấy top k. nhưng vì documents là các trang PDF, nên k=15 là hợp lý, không quá nhiều để tránh noise.
    bm25_retriever.k = k
    # từ vectorstore, tạo vector retriever, vector retriever sẽ tìm kiếm các đoạn có ngữ nghĩa liên quan
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )


def seteup_genai_client() -> genai.Client:
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY chưa được cấu hình trong .env")
    return genai.Client(api_key=settings.GEMINI_API_KEY)


client = seteup_genai_client()


def build_rag_chain(retriever):
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        google_api_key=settings.GEMINI_API_KEY,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "NGỮ CẢNH:\n{context}\n\nCâu hỏi: {question}\n\nTrả lời:"),
    ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ─────────────── PIPELINE ───────────────

def index_pipeline(pdf_path: str = PDF_PATH, use_hybrid: bool = True):
    print("[1/4] Đang load PDF...")
    docs = load_pdf(pdf_path)
    print(f"  → {len(docs)} trang")

    print("[2/4] Đang split documents (cấu trúc, chunk=512/128)...")
    chunks = split_documents_v1(docs)
    print(f"  → {len(chunks)} chunks")

    print("[3/4] Đang tạo embeddings và lưu Chroma (rate-limited)...")
    vectorstore = create_vectorstore(chunks)
    print(f"  → Đã lưu tại {CHROMA_DIR}")

    if use_hybrid:
        print("[4/4] Xây dựng RAG chain với hybrid retriever (BM25+Vector)...")
        retriever = build_hybrid_retriever(vectorstore, chunks, k=HYBRID_K)
    else:
        print("[4/4] Xây dựng RAG chain với vector retriever...")
        retriever = build_vector_retriever(vectorstore, k=RETRIEVER_K)

    chain = build_rag_chain(retriever)

    print("\n>>> Test 3 câu hỏi:")
    test_queries(chain)

    return chain


# ─────────────── CHAT (KHÔNG RE-INDEX) ───────────────

_chain = None
_chunks = None


def _ensure_chain(k: int = RETRIEVER_K, use_hybrid: bool = True):
    global _chain, _chunks
    if _chain is None:
        vstore = load_vectorstore()
        if use_hybrid:
            if _chunks is None:
                docs = load_pdf()
                _chunks = split_documents_v1(docs)
            retriever = build_hybrid_retriever(vstore, _chunks, k=HYBRID_K)
        else:
            retriever = build_vector_retriever(vstore, k=k)
        _chain = build_rag_chain(retriever)
    return _chain


def chat(question: str, k: int = RETRIEVER_K) -> str:
    chain = _ensure_chain(k=k, use_hybrid=True)
    return ask(chain, question)


def reset_chain():
    global _chain, _chunks
    _chain = None
    _chunks = None


def test_queries(chain=None):
    if chain is None:
        chain = _ensure_chain()
    queries = [
        # "Hãy hướng dẫn kết nối Wifi cho điện thoại Samsung.",
        # "Điện thoại Samsung của tôi bị treo logo, làm thế nào để khắc phục?",
        # "Giá bán của sản phẩm này là bao nhiêu?",
        "Mô tả sản phẩm có mã SM-A125F?"
    ]
    for q in queries:
        print(f"\n{'='*60}")
        print(f"Câu hỏi: {q}")
        print(f"Trả lời: {ask(chain, q)}")
    print()


def compare_retrievers():
    print(f"\n{'='*60}")
    print("CHUẨN BỊ DỮ LIỆU CHO COMPARISON")
    print(f"{'='*60}")
    print("[1/4] Đang load PDF...")
    docs = load_pdf()
    print(f"  → {len(docs)} trang")
    print("[2/4] Đang split documents...")
    chunks = split_documents_v1(docs)
    print(f"  → {len(chunks)} chunks")
    print("[3/4] Đang tạo embeddings và lưu Chroma...")
    vectorstore = load_vectorstore()
    print(f"  → Đã lưu tại {CHROMA_DIR}")

    all_results = {}

    for variant_name, build_fn in [
        ("vector_only", lambda: build_vector_retriever(vectorstore, k=RETRIEVER_K)),
        ("hybrid", lambda: build_hybrid_retriever(vectorstore, chunks, k=HYBRID_K)),
    ]:
        print(f"\n{'='*60}")
        print(f"EVALUATING: {variant_name}")
        print(f"{'='*60}")
        retriever = build_fn()
        chain = build_rag_chain(retriever)
        result = run_evaluation(variant_name, retriever, chain, TEST_CASES)
        all_results[variant_name] = result

        kw_str = f"{result['keyword_coverage']}" if result['keyword_coverage'] is not None else "N/A"
        print(f"\n>>> {variant_name} SUMMARY:")
        print(f"    Context Recall : {result['context_recall']}")
        print(f"    Keyword Coverage: {kw_str}")
        print(f"    Avg Latency    : {result['avg_latency']}s")

    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Variant':14s} {'Context Rec':>12s} {'Keyword Cov':>12s} {'Avg Latency':>12s}")
    print(f"{'─'*14} {'─'*12} {'─'*12} {'─'*12}")
    for name, res in all_results.items():
        kw = f"{res['keyword_coverage']:.4f}" if res['keyword_coverage'] is not None else "N/A"
        print(f"{name:14s} {str(res['context_recall']):>12s} {kw:>12s} {str(res['avg_latency'])+'s':>12s}")
    print(f"{'='*60}\n")

    save_report(all_results, "comparison_report.json")


def ask(chain, question: str) -> str:
    """Hàm hỏi đáp RAG, trả về câu trả lời từ LLM."""
    return chain.invoke(question)

def print_chunk_detail(chunks, label: str, max_print: int = 5):
    print(f"\n{'='*60}")
    print(f"{label}: {len(chunks)} chunks")
    print(f"{'='*60}")
    for i, d in enumerate(chunks):
        page = d.metadata.get("page", "?")
        idx = d.metadata.get("start_index", "?")
        print(f"\n--- Chunk {i+1}/{len(chunks)} (page={page}, start_index={idx}) ---")
        print(d.page_content)
    # if len(chunks) > max_print:
    #     print(f"\n... và {len(chunks) - max_print} chunks khác")
    
def test_chunk_num():
    """Test số lượng chunk tạo ra từ 2 cấu trúc chunk_size khác nhau"""
    docs = load_pdf()
    chunks1 = split_documents_v1(docs, chunk_size=512, chunk_overlap=128)
    chunks2 = split_documents_v1(docs, chunk_size=1000, chunk_overlap=200)
    print(f"Chunk 512/128: {len(chunks1)} chunks")
    print(f"Chunk 1000/200: {len(chunks2)} chunks")

    print_chunk_detail(chunks1, "DETAIL 512/128", max_print=5)
    # print_chunk_detail(chunks2, "DETAIL 1000/200", max_print=5)
    
if __name__ == "__main__":
    if not settings.GEMINI_API_KEY:
        print("Lỗi: GEMINI_API_KEY chưa được cấu hình trong .env")
    else:
        # index_pipeline(use_hybrid=True)
        # compare_retrievers()
        test_chunk_num()
