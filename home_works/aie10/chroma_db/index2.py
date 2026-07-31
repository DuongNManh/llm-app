import json
import time
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GEMINI_API_KEY: str | None = None
    model_config = {
        "env_file": ".env", "extra": "ignore"}
    
    
settings = Settings()

SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia hỗ trợ kỹ thuật và chăm sóc khách hàng của sản phẩm Samsung Smart Phone. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng bằng cách DỰA TRÊN NGỮ CẢNH được cung cấp từ tài liệu. Hãy tuân thủ nghiêm ngặt các yêu cầu sau:
1. Chỉ trả lời dựa trên thông tin trong đoạn trích. không suy đoán hoặc thêm thông tin bên ngoài.
2. Nếu đoạn trích không đủ thông tin để trả lời, hãy nói "Tài liệu không đề cập đến vấn đề này."
3. Trả lời theo nội dung trích dẫn, cùng với toàn bộ đoạn trích (nếu có) chứa câu trả lời.
4. Trả lời bằng tiếng Việt, thân thiện và dễ hiểu, không sử dụng từ ngữ chuyên ngành quá khó hiểu."""

PDF_PATH = "documents/huongdansudungSamSung.pdf"
CHROMA_DIR = "./chroma_db_v3"
EMBED_BATCH_SIZE = 10
EMBED_BATCH_SLEEP = 10
RETRIEVER_K = 4
HYBRID_K = 15


# ─────────────── TEST CASES ───────────────

from pydantic import BaseModel

class TestCase(BaseModel):
    id : int
    question : str
    expected_answer_contains : list[str]
    expected_source_pages : list[int]
    category: str

TEST_CASES = [
    TestCase(id=1, question="Hãy hướng dẫn kết nối Wifi cho điện thoại Samsung.",
             expected_answer_contains=["Cài đặt", "Kết nối", "Wi-Fi", "mạng Wi-Fi"], expected_source_pages=[65, 66], category="general"),
    TestCase(id=2, question="Cách chụp màn hình trên điện thoại Samsung?",
             expected_answer_contains=["phím Cạnh", "Giảm âm lượng", "cùng lúc"], expected_source_pages=[30], category="general"),
    TestCase(id=3, question="Pin điện thoại Samsung nên sạc như thế nào cho đúng?",
             expected_answer_contains=["sạc", "pin", "USB Type-C"], expected_source_pages=[13, 14], category="general"),
    TestCase(id=4, question="Làm thế nào để chuyển dữ liệu từ máy cũ sang máy Samsung mới?",
             expected_answer_contains=["Smart Switch", "dữ liệu", "chuyển"], expected_source_pages=[21, 22], category="general"),
    TestCase(id=5, question="Làm sao để vào Internet qua Wifi trên máy Samsung?",
             expected_answer_contains=["Cài đặt", "Kết nối", "Wi-Fi"], expected_source_pages=[65, 66], category="semantic_gap"),
    TestCase(id=6, question="Điện thoại Samsung bị treo logo, làm thế nào để khắc phục?",
             expected_answer_contains=["nhấn và giữ", "phím Cạnh", "phím Giảm âm lượng", "7 giây", "khởi động lại"], expected_source_pages=[19], category="semantic_gap"),
    TestCase(id=7, question="Máy Samsung của tôi bị đơ, không bấm được gì, phải làm sao?",
             expected_answer_contains=["nhấn và giữ", "phím Cạnh", "phím Giảm âm lượng", "khởi động lại"], expected_source_pages=[19], category="semantic_gap"),
    TestCase(id=8, question="Samsung của tôi bị nóng quá, có sao không?",
             expected_answer_contains=["nóng", "thiết bị", "sạc", "ứng dụng"], expected_source_pages=[6, 7, 8], category="semantic_gap"),
    TestCase(id=9, question="Làm cách nào sao chép ảnh từ Samsung qua máy vi tính?",
             expected_answer_contains=["Smart Switch", "máy tính", "dữ liệu"], expected_source_pages=[22], category="semantic_gap"),
    TestCase(id=10, question="SM-A125F/DS dùng loại thẻ SIM nào?",
             expected_answer_contains=["nano SIM", "SIM"], expected_source_pages=[15, 16], category="code_model"),
    TestCase(id=11, question="Điện thoại Samsung có hỗ trợ Dolby Atmos không?",
             expected_answer_contains=["Dolby Atmos", "âm thanh", "Cài đặt"], expected_source_pages=[72], category="code_model"),
    TestCase(id=12, question="Smart Switch có thể chuyển dữ liệu bằng cách nào?",
             expected_answer_contains=["Smart Switch", "Không dây", "máy tính", "dữ liệu"], expected_source_pages=[21, 22], category="code_model"),
    TestCase(id=13, question="Samsung Members giúp ích gì khi máy gặp vấn đề?",
             expected_answer_contains=["Samsung Members", "hỗ trợ", "chẩn đoán"], expected_source_pages=[56], category="code_model"),
    TestCase(id=14, question="Giá bán của sản phẩm này là bao nhiêu?",
             expected_answer_contains=["Tài liệu không đề cập", "không đề cập"], expected_source_pages=[], category="out_of_scope"),
    TestCase(id=15, question="Bảo hành điện thoại Samsung bao lâu?",
             expected_answer_contains=["Tài liệu không đề cập", "không đề cập"], expected_source_pages=[], category="out_of_scope"),
]



## =========== Pipeline indexing ============= 
# thực hiện load_pdf -> split_document -> create_embeddings -> create_vectorstore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path: str = PDF_PATH) -> list:
    """Load PDF và trả về các trang dưới dạng Document, có meta data (page)"""
    # loader = PyMuPDFLoader(file_path, extract_tables="markdown")
    # mode  extraction_mode="layout" có gì khác với mode extraction_mode="plain"?
    loader = PyMuPDFLoader(file_path)
    return loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_document_v1(docs: list, chunk_size = 512, chunk_overlap = 128) -> list:
    """Split các trang Document thành các đoạn chunk nhỏ hơn để tạo embedding
        Separators ưu tiên: headers -> paragraphs -> sentences"""
    # Trong bài excersize này, em sẽ tạo 2 comparision từ (1000/200) và (512/128)
    # để kiểm thử chunk_size khác thì thế nào?
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        add_start_index = True,
        separators=[
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )
    return splitter.split_documents(docs)

from langchain_google_genai import GoogleGenerativeAIEmbeddings

def create_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Tạo embedding model từ Google Embedding API"""
    gemini_api_key = settings.GEMINI_API_KEY
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY chưa được cấu hình")
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        api_key=gemini_api_key
    )



CHUNK1000va200 = "chunk1000va200"
CHUNK512va128 = "chunk512va128"

from langchain_chroma import Chroma

def create_vectorstore(
    documents: list,
    per_dir: str = CHROMA_DIR,
    collection_name : str = CHUNK512va128,
    batch_size: int = EMBED_BATCH_SIZE,
    batch_sleep: int = EMBED_BATCH_SLEEP
) -> Chroma:
    
    """tạo vector store lưu các embedding"""
    embeddings = create_embeddings()
    # Em có để collection_name để lát comparision giữa 2 chunksize
    vector_store = Chroma(
        embedding_function= embeddings,
        persist_directory= per_dir,
        collection_name= collection_name
    )
    # batch embedding để tránh rate limit của google
    total = len(documents)
    for i in range(0, total, batch_size):
        batch = documents[i : i + batch_size]
        vector_store.add_documents(batch)
        print(f" => batch {i//batch_size + 1}/{(total-1)//batch_size + 1}: {len(batch)}")
        if i + batch_size < total:
            time.sleep(batch_sleep)
    
    return vector_store


def load_vectorstore(
    per_dir: str = CHROMA_DIR,
    collection_name : str = CHUNK512va128) -> Chroma:
    
    """load lại vector store đã persite (theo collection name)"""
    # collection_name để load collection giữa 2 chunk size
    embeddings = create_embeddings()
    return Chroma(
        embedding_function=embeddings,
        persist_directory=per_dir,
        collection_name=collection_name
    )


## ============= Evaluation Metrics ===============

def normalize_text(text: str) -> str:
    return text.lower().strip()


def keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    """tỉ lệ các keyword đã cover"""
    if not expected_keywords:
        return 1.0
    # normalize anser đi để dễ compare
    answer_lower = normalize_text(answer)
    matchs = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return matchs / len(expected_keywords)


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
    # trả về bool nếu có page trùng xuất hiện 
    # trong retrieved_page và expected_page (test case)
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
            status = "OK" if (r["context_recall"] if r["expected_pages"] else True) else "NG"
            lines.append(f"#{r["id"]}  [{status}]  {r['category']}  CR={r['context_recall']}  KW={r['keyword_coverage']:.2f}  {r['latency_total']:.1f}s")
            lines.append(f"     pages => {r['retrieved_pages']}  (expected {r['expected_pages']})")
            lines.append(f"     Q: {r['question']}")
            lines.append(f"     A: {r['answer_preview']}")
            lines.append("")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"    => Saved {path} and {txt_path}")

def run_evaluation(name: str, retriever, chain, test_cases: list) -> dict:
    """hàm chạy và compare giữa các method"""
    
    results = []
    total_cr = total_latency = 0
    total_kw = 0.0
    cr_count = kw_count = 0
    
    for tc in test_cases:
        question = tc.question
        expected_kw = tc.expected_answer_contains
        expected_pages = tc.expected_source_pages
        
        t0 = time.time()
        retrieved_docs = retriever.invoke(question)
        answer = chain.invoke(question)
        print(f"\n>>> Test case #{tc.id} | {tc.category} | Q: {question}")
        print(f"  → Retrieved {len(retrieved_docs)} docs, Answer: {answer[:100]}...")
        t2 = time.time()
        
        # kiểm tra metric context recall và keyword coverage
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
            "id": tc.id,
            "category": tc.category,
            "question": question,
            "answer_preview": answer,
            "context_recall": cr,
            "keyword_coverage": round(kw, 4),
            "latency_total": round(t2 - t0, 3),
            "retrieved_pages": retrieved_page_list,
            "expected_pages": expected_pages
        })
        
        # kiểm tra expected page để làm gì?
        # nếu expected page có, thì mới tính trung bình context recall
        # giải thích từng biến:
        # total_cr: tổng số test case có context recall = True
        # cr_count: tổng số test case có expected page (có thể tính trung bình context recall)
        if expected_pages:
            total_cr += int(cr) 
            cr_count += 1
        
        # kiểm tra expected keyword để làm gì?
        # nếu expected keyword có, thì mới tính trung bình keyword coverage
        # giải thích từng biến:
        # total_kw: tổng số keyword coverage (tỉ lệ từ 0 đến 1)
        # kw_count: tổng số test case có expected keyword (có thể tính trung bình keyword coverage)
        if expected_kw:
            total_kw += kw
            kw_count += 1
            
        total_latency += t2 - t0
        status = "✓" if (cr if expected_pages else True) else "✗"
        print(f"  [{status}] #{tc.id} {tc.category:14s} | CR={cr} KW={kw:.2f} | pages={retrieved_page_list} | {t2-t0:.1f}s")
    
    return {
        "variant": name,
        "num_cases": len(test_cases),
        "context_recall": round(total_cr / cr_count, 4) if cr_count > 0 else None,
        "keyword_coverage": round(total_kw / kw_count, 4) if kw_count > 0 else None,
        "avg_latency": round(total_latency / len(test_cases), 3) if test_cases else None,
        "details": results
    }


## ================ Setup retriever và chain ===============

def format_docs(docs: list) -> str:
    """format các Document thành str để đưa vào prompt"""
    print(f"\n >>> Retrieved  {len(docs)} chunks")
    for i, d in enumerate(docs):
        print(f"  [{i}] page={d.metadata.get('page')} | {d.page_content[:100]}...")
    return "\n\n---\n\n".join(
        f"[Tài liệu {doc.metadata.get('source', 'Unknown')}] [Trang {doc.metadata.get('page', '?')}] {doc.page_content[:100]}..."
        for doc in docs
    )

from langchain_core.vectorstores import VectorStoreRetriever

def build_vector_retriever(vectorstore, k: int = RETRIEVER_K) -> VectorStoreRetriever:
    """retriever search thuần vector (baseline)"""
    return vectorstore.as_retriever(search_kwargs={"k": k})

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

def     build_hybrid_retriever(vectorstore : Chroma, documents: list, k: int = HYBRID_K, k_vector: int = 3) -> EnsembleRetriever:
    """Hybrid retriver: BM25 (keyword) + Vector (semantic).
    
    Kỹ thuật cải tiến chính:
    - BM25 bắt keyword chính xác trong documents gốc (không phải chunk)
    - Vector search tìm kiếm semantic trong các chunk đã được embedding
    """

    # từ documents gốc, tạo BM25 retriever, BM25 sẽ tìm kiếm các đoạn có từ khóa liên quan
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k 
    # từ vectorstore, tạo vector retriever
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k_vector})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain_core.runnables import Runnable


class RerankBM25Retriever(Runnable[str, list[Document]]):
    """BM25 → CrossEncoder rerank → top-k (LangChain Runnable).

    - BM25 retrieve k_retrieve chunks (nhiều, bắt keyword)
    - CrossEncoder rerank: tính relevance score giữa query và từng chunk
    - Giữ lại k_final chunks có điểm cao nhất (giảm noise)
    """
    def __init__(self, documents: list, reranker: CrossEncoder, k_retrieve: int = 15, k_final: int = 4):
        from langchain_community.retrievers import BM25Retriever
        self.bm25 = BM25Retriever.from_documents(documents)
        self.bm25.k = k_retrieve
        self.reranker = reranker
        self.k_final = k_final
        self.k_retrieve = k_retrieve

    def invoke(self, input: str, config=None) -> list[Document]:
        docs = self.bm25.invoke(input, config=config)
        if not docs or len(docs) <= self.k_final:
            return docs[:self.k_final]
        pairs = [(input, d.page_content) for d in docs]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        scored = list(zip(scores, docs))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:self.k_final]]


def build_reranked_hybrid_retriever(
    vectorstore: Chroma, documents: list,
    k_bm25: int = 15, k_final: int = 4, k_vector: int = 3,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> EnsembleRetriever:
    """Hybrid retriever với rerank: BM25 → CrossEncoder → top-k → ensemble với Vector.

    - BM25 retrieve rộng (k_bm25) → rerank giữ lại k_final chunk chính xác nhất
    - Vector search lấy k_vector chunk (ngữ nghĩa)
    - Ensemble weighted fusion
    - Kết quả cuối: ~k_final + k_vector chunk (đã lọc noise)
    """
    print(f"  → Loading reranker model '{reranker_model}' (first run downloads ~80MB)...")
    reranker = CrossEncoder(reranker_model)
    bm25_reranked = RerankBM25Retriever(documents, reranker, k_retrieve=k_bm25, k_final=k_final)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k_vector})

    return EnsembleRetriever(
        retrievers=[bm25_reranked, vector_retriever],
        weights=[0.6, 0.4],
    )


from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

def debug_chain_question_context(inputs):
    """Debug chain: in ra question và context"""
    question = inputs.get("question")
    context = inputs.get("context")
    print(f"\n>>> QUESTION: {question}")
    print(f"\n>>> CONTEXT: {context[:250]}...")
    return inputs

def build_rag_chain(retriever: EnsembleRetriever | VectorStoreRetriever, system_prompt: str = SYSTEM_PROMPT):
    """tạo RAG chain với prompt template"""

    gemini_api_key = settings.GEMINI_API_KEY
    llm = ChatGoogleGenerativeAI(
        model = "gemini-3.1-flash-lite",
        gemini_api_key = gemini_api_key,
        temperature = 0.0,
    )

    # tạo prompt template với system prompt và human prompt
    # tương tự như pure template gọi gemini thay vì sử dụng ChatGoogleGenerativeAI
    # ví dụ cho pure template, ko dùng ChatPromptTemplate:
    # [{"role": "system", "content": system_prompt}, {"role": "user", "content": "NGỮ CẢNH:\n{question}\n\nCâu hỏi: {context}\n\n Trả lời:"}]
    # llm = google_genai.Client(model="gemini-3.1-flash-lite", gemini_api_key=gemini_api_key)
    # response = llm.chat([{"role": "system", "content": system_prompt}, {"role": "user", "content": f"NGỮ CẢNH:\n{question}\n\nCâu hỏi: {context}\n\n Trả lời:"}])
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "NGỮ CẢNH:\n{question}\n\nCâu hỏi: {context}\n\n Trả lời:"),
    ])

    # tạo chain: retriever -> format_docs -> prompt_template -> llm -> StrOutputParser
    # biến context: là ngữ cảnh được format từ các Document đã retrieve và sử dụng format_docs để biến thành str
    # biến question: là câu hỏi của người dùng
    # prompt_template sẽ nhận 2 biến context và question để tạo prompt cho llm
    # llm sẽ trả về str, và StrOutputParser sẽ parse str thành output cuối cùng
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RunnableLambda(debug_chain_question_context)
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return chain


##============ implement indexing pipeline =============

def indexing_pipeline(pdf_path: str = PDF_PATH, chroma_dir: str = CHROMA_DIR, collection_name: str = CHUNK512va128, use_hybrid: bool = True):
    """pipeline indexing: load_pdf -> split_document -> create_embeddings -> create_vectorstore"""
    print(f"[1/4] Đang load PDF {pdf_path}...")
    docs = load_pdf(pdf_path)
    print(f"  → {len(docs)} trang")

    print("[2/4] Đang split documents (cấu trúc, chunk=512/128)...")
    chunks = split_document_v1(docs)
    print(f"  → {len(chunks)} chunks")

    print("[3/4] Đang tạo embeddings và lưu Chroma (rate-limited)...")
    vectorstore = create_vectorstore(chunks, per_dir=chroma_dir, collection_name=collection_name)
    print(f"  → Lưu Chroma tại {chroma_dir}/{collection_name}")

    if use_hybrid:
        print("[4/4] Đang tạo hybrid retriever (BM25 + Vector)...")
        retriever = build_hybrid_retriever(vectorstore, docs)
        print("  → Hybrid retriever đã sẵn sàng.")
    else:
        print("[4/4] Đang tạo vector retriever (Vector only)...")
        retriever = build_vector_retriever(vectorstore)
        print("  → Vector retriever đã sẵn sàng.")

    # langchain chain sử dụng retriever
    chain = build_rag_chain(retriever)

    # test queries
    print(f"\n{'='*60}")
    print("TEST QUERIES")
    test_queries(chain)


def test_queries(chain):
    if chain is None:
        print("Chưa có chain, vui lòng tạo chain trước khi test queries.")
        return
    queries = [
        "Hãy hướng dẫn kết nối Wifi cho điện thoại Samsung.",
        "Điện thoại Samsung của tôi bị treo logo, làm thế nào để khắc phục?",
        "Giá bán của sản phẩm này là bao nhiêu?",
        "Mô tả sản phẩm có mã SM-A125F?"
    ]
    for q in queries:
        print(f"\n{'='*60}")
        print(f"Câu hỏi: {q}")
        print(f"Trả lời: {ask(chain, q)}")
    print()
    
def ask(chain, question: str) -> str:
    """Hàm hỏi đáp RAG, trả về câu trả lời từ LLM."""
    return chain.invoke(question)

# ============ COMPARE và in báo cáo =============
from pathlib import Path

def compare_retrievers():
    print(f"\n{'='*60}")
    print("CHUẨN BỊ DỮ LIỆU CHO COMPARISON")
    print(f"{'='*60}")
    print("[1/4] Đang load PDF...")
    docs = load_pdf()
    print(f"  → {len(docs)} trang")
    print("[2/4] Đang split documents cấu trúc 512/128...")
    chunks1 = split_document_v1(docs)
    print(f"  → {len(chunks1)} chunks")
    print("[2/4] Đang split documents cấu trúc 1000/200...")
    chunks2 = split_document_v1(docs, chunk_size=1000, chunk_overlap=200)
    print(f"  → {len(chunks2)} chunks")
    print("[3/4] Đang tạo embeddings và lưu Chroma...")
    
    # nếu chưa có vectorstore, thì tạo mới, nếu đã có thì load lại
    if not Path(CHROMA_DIR).exists():
        print(f"  → Chưa có Chroma, tạo mới tại {CHROMA_DIR}...")
        vectorstore_1000_200 = create_vectorstore(chunks2, collection_name="chunk1000va200")
        vectorstore_512_128 = create_vectorstore(chunks1, collection_name="chunk512va128")
    else:
        print(f"  → Chroma đã tồn tại, load lại từ {CHROMA_DIR}...")
        vectorstore_1000_200 = load_vectorstore(collection_name="chunk1000va200")
        vectorstore_512_128 = load_vectorstore(collection_name="chunk512va128")
        if vectorstore_1000_200 is None or vectorstore_512_128 is None:
            print(f"  → Không tìm thấy collection, tạo mới tại {CHROMA_DIR}...")
            vectorstore_1000_200 = create_vectorstore(chunks2, collection_name="chunk1000va200")
            vectorstore_512_128 = create_vectorstore(chunks1, collection_name="chunk512va128")
    print(f"  → Hoàn thành tạo/load Chroma tại {CHROMA_DIR}")

    all_results = {}

    for variant_name, build_fn in [
        ("vector_only_1000/200", lambda: build_vector_retriever(vectorstore_1000_200, k=RETRIEVER_K)),
        ("vector_only_512/128", lambda: build_vector_retriever(vectorstore_512_128, k=RETRIEVER_K)),
        ("hybrid", lambda: build_hybrid_retriever(vectorstore_512_128, chunks1, k=HYBRID_K)),
        ("reranked_hybrid", lambda: build_reranked_hybrid_retriever(vectorstore_512_128, chunks1, k_bm25=10, k_final=4, k_vector=3)),
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
        print(f"    Total Cases    : {result['num_cases']}")
        save_report({variant_name: result}, f"report_{variant_name}.json")
        time.sleep(30)
        print("")
        print("rate limit: chờ 30s trước khi chạy variant tiếp theo...")
        print(f"\n{'='*60}\n\n")

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
    
    
## =========== Debug chunk number =============

def print_chunk_detail(chunks, label: str, max_print: int = 5):
    print(f"\n{'='*60}")
    print(f"{label}: {len(chunks)} chunks")
    print(f"{'='*60}")
    for i, d in enumerate(chunks):
        page = d.metadata.get("page", "?")
        idx = d.metadata.get("start_index", "?")
        print(f"\n--- Chunk {i+1}/{len(chunks)} (page={page}, start_index={idx}) ---")
        print(d.page_content)
    if len(chunks) > max_print:
        print(f"\n... và {len(chunks) - max_print} chunks khác")
        break


def test_chunk_num():
    """Test số lượng chunk tạo ra từ 2 cấu trúc chunk_size khác nhau"""
    docs = load_pdf()
    chunks1 = text_spliter(docs, chunk_size=512, chunk_overlap=128)
    chunks2 = text_spliter(docs, chunk_size=1000, chunk_overlap=200)
    print(f"Chunk 512/128: {len(chunks1)} chunks")
    print(f"Chunk 1000/200: {len(chunks2)} chunks")

    print_chunk_detail(chunks1, "DETAIL 512/128", max_print=5)
    print_chunk_detail(chunks2, "DETAIL 1000/200", max_print=5)

##=================== MAIN ==================


    
if __name__ == "__main__":
    if not settings.GEMINI_API_KEY:
        print("Vui lòng cấu hình GEMINI_API_KEY trong .env trước khi chạy.")
    else:
        # chỉ indexing pipeline
        indexing_pipeline(pdf_path=PDF_PATH, chroma_dir=CHROMA_DIR, collection_name=CHUNK512va128, use_hybrid=True)
        # compare metric các phương pháp và in báo cáo
        compare_retrievers()
        # test_queries(build_rag_chain(build_reranked_hybrid_retriever(load_vectorstore(collection_name=CHUNK512va128), load_pdf(), k_bm25=10, k_final=4, k_vector=3)))
        # test_chunk_num()