import json
import time
from pathlib import Path
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder
from langchain_classic.retrievers import EnsembleRetriever


class Settings(BaseSettings):
    GEMINI_API_KEY: str | None = None
    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
if not settings.GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in environment variables or .env file.")

SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia hỗ trợ kỹ thuật và chăm sóc khách hàng của sản phẩm Samsung Smart Phone. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng bằng cách DỰA TRÊN NGỮ CẢNH được cung cấp từ tài liệu. Hãy tuân thủ nghiêm ngặt các yêu cầu sau:
1. Chỉ trả lời dựa trên thông tin trong đoạn trích. không suy đoán hoặc thêm thông tin bên ngoài.
2. Nếu đoạn trích không đủ thông tin để trả lời, hãy nói "Tài liệu không đề cập đến vấn đề này."
3. Trả lời theo nội dung trích dẫn, cùng với toàn bộ đoạn trích (nếu có) chứa câu trả lời.
4. Trả lời bằng tiếng Việt, thân thiện và dễ hiểu, không sử dụng từ ngữ chuyên ngành quá khó hiểu."""

PDF_PATH = "documents/huongdansudungSamSung.pdf"
CHROMA_DIR = "./home_works/aie10/chroma_db_v4"
EMBED_BATCH_SIZE = 10
EMBED_BATCH_SLEEP = 10
RETRIEVER_K = 4
HYBRID_K = 15


# ═══════════════════════════════════════════════════════════════
# 2. Data models
# ═══════════════════════════════════════════════════════════════


class TestCase(BaseModel):
    id: int
    question: str
    expected_answer_contains: list[str]
    expected_source_pages: list[int]
    category: str


class CaseResult(BaseModel):
    id: int
    category: str
    question: str
    answer_preview: str
    context_recall: bool
    keyword_coverage: float
    latency_total: float
    retrieved_pages: list
    expected_pages: list[int]
    expected_keywords: list[str]
    keywords_found: list[str]


class VariantResult(BaseModel):
    variant: str
    num_cases: int
    context_recall: float | None = None
    keyword_coverage: float | None = None
    avg_latency: float | None = None
    details: list[CaseResult]


TEST_CASES = [
    TestCase(id=1, question="Hãy hướng dẫn kết nối Wifi (Wi-Fi) cho điện thoại Samsung.",
             expected_answer_contains=["Cài đặt", "Kết nối", "Wi-Fi", "mạng Wi-Fi"], expected_source_pages=[65, 66], category="general"),
    # TestCase(id=2, question="Cách chụp màn hình trên điện thoại Samsung?",
    #          expected_answer_contains=["phím Cạnh", "Giảm âm lượng", "cùng lúc"], expected_source_pages=[30], category="general"),
    TestCase(id=3, question="Pin điện thoại Samsung nên sạc như thế nào cho đúng?",
             expected_answer_contains=["sạc", "pin", "USB Type-C", "phụ kiện chính hãng"], expected_source_pages=[13, 14], category="general"),
    TestCase(id=4, question="Làm thế nào để chuyển dữ liệu từ máy cũ sang máy Samsung mới?",
             expected_answer_contains=["Smart Switch", "dữ liệu", "chuyển"], expected_source_pages=[21, 22], category="general"),
    TestCase(id=5, question="Làm sao để vào Internet qua Wifi trên máy Samsung?",
             expected_answer_contains=["Cài đặt", "Kết nối", "Wi-Fi"], expected_source_pages=[65, 66], category="semantic_gap"),
    TestCase(id=6, question="Điện thoại Samsung bị treo logo, làm thế nào để khắc phục?",
             expected_answer_contains=["nhấn và giữ", "phím Cạnh", "phím Giảm âm lượng", "7 giây", "khởi động lại"], expected_source_pages=[19], category="semantic_gap"),
    # TestCase(id=7, question="Máy Samsung của tôi bị đơ, không bấm được gì, phải làm sao?",
    #          expected_answer_contains=["nhấn và giữ", "phím Cạnh", "phím Giảm âm lượng", "khởi động lại"], expected_source_pages=[19], category="semantic_gap"),
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


# ═══════════════════════════════════════════════════════════════
# 3. Document loading & splitting
# ═══════════════════════════════════════════════════════════════


def load_pdf(pdf_path: str = PDF_PATH) -> list[Document]:
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def split_documents(docs: list[Document], chunk_size: int = 512, chunk_overlap: int = 128) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        keep_separator=True,
        separators=["\n\n", "\n", ".", "?", "!", ",", " ", ""],
    )
    return splitter.split_documents(docs)


# ═══════════════════════════════════════════════════════════════
# 4. Vector store management
# ═══════════════════════════════════════════════════════════════


def _create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cpu"},
    )

class VectorStoreManager:
    def __init__(self, persist_directory: str = CHROMA_DIR, collection_name: str = "chunk512va128"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = _create_embeddings()
        self.vectorstore: Chroma | None = None

    def create(self, documents: list[Document]) -> Chroma:
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self.vectorstore.persist()
        print(f"  => Created vectorstore: {len(documents)} docs at {self.persist_directory}")
        return self.vectorstore

    def load(self) -> Chroma:
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            collection_metadata={"hnsw:space": "cosine"},
        )
        print(f"  => Loaded vectorstore from {self.persist_directory}/{self.collection_name}")
        return self.vectorstore


# ═══════════════════════════════════════════════════════════════
# 5. Retriever variants
# ═══════════════════════════════════════════════════════════════


def build_vector_retriever(vectorstore: Chroma, k: int = RETRIEVER_K) -> VectorStoreRetriever:
    """Semantic search thuần — baseline."""
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_ensemble_hybrid_retriever(
    vectorstore: Chroma,
    documents: list[Document],
    k_vector: int = 5,
    k_bm25: int = 10,
    weight_vector: float = 0.5,
    weight_bm25: float = 0.5,
) -> EnsembleRetriever:
    """Hybrid: BM25 + Vector, weighted ensemble fusion (không rerank)."""
    vector_ret = vectorstore.as_retriever(search_kwargs={"k": k_vector})
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = k_bm25
    return EnsembleRetriever(
        retrievers=[bm25, vector_ret],
        weights=[weight_bm25, weight_vector],
    )


# ═══════════════════════════════════════════════════════════════
# 6. CrossEncoder reranker
# ═══════════════════════════════════════════════════════════════

_global_reranker = CrossEncoder("BAAI/bge-reranker-large", device="cpu", max_length=512)

class CrossEncoderReranker:
    """Stateless reranker: nhận query + docs, trả về top-k docs đã rerank."""

    def __init__(self, model: CrossEncoder, top_k: int = 4, batch_size: int = 15):
        self.model = model
        self.top_k = top_k
        self.batch_size = batch_size

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        if len(docs) <= self.top_k:
            return docs

        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs, show_progress_bar=False, batch_size=self.batch_size)

        for i, (score, doc) in enumerate(zip(scores, docs)):
            print(f"    [{i}] score={score:.4f} | page={doc.metadata.get('page')} | {doc.page_content[:100]}...")

        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

        result = []
        for score, doc in ranked[: self.top_k]:
            doc.metadata["relevance_score"] = float(score)
            print(f"    [TOP] score={score:.4f} | page={doc.metadata.get('page')} | {doc.page_content[:100]}...")
            result.append(doc)
        return result


class HybridRerankRetriever(Runnable[str, list[Document]]):
    """BM25 + Vector → dedup → CrossEncoder rerank → top-k.

    Injectable retriever: có thể dùng trực tiếp trong LangChain pipeline.
    """

    def __init__(
        self,
        documents: list[Document],
        vectorstore: Chroma,
        reranker: CrossEncoderReranker,
        k_bm25: int = 15,
        k_vector: int = 10,
    ):
        self.bm25 = BM25Retriever.from_documents(documents)
        self.bm25.k = k_bm25
        self.vector_ret = vectorstore.as_retriever(search_kwargs={"k": k_vector})
        self.reranker = reranker

    @staticmethod
    def _deduplicate(docs: list[Document]) -> list[Document]:
        seen = set()
        unique = []
        for doc in docs:
            key = doc.metadata.get("id") or doc.metadata.get("source") or doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique

    def invoke(self, input: str | dict, config=None, **kwargs) -> list[Document]:
        query = input.get("question") or input.get("input") or str(input) if isinstance(input, dict) else str(input)

        bm25_docs = self.bm25.invoke(query, config=config)
        vector_docs = self.vector_ret.invoke(query, config=config)
        candidates = self._deduplicate(bm25_docs + vector_docs)
        return self.reranker.rerank(query, candidates)


def build_reranked_hybrid_retriever(
    vectorstore: Chroma,
    documents: list[Document],
    k_bm25: int = 15,
    k_vector: int = 10,
    top_k: int = 4,
    reranker: CrossEncoderReranker | None = None,
) -> HybridRerankRetriever:
    """Hybrid: BM25 + Vector → dedup → CrossEncoder rerank → top-k."""
    if reranker is None:
        reranker = CrossEncoderReranker(model=_global_reranker, top_k=top_k)
    return HybridRerankRetriever(
        documents=documents,
        vectorstore=vectorstore,
        reranker=reranker,
        k_bm25=k_bm25,
        k_vector=k_vector,
    )


# ═══════════════════════════════════════════════════════════════
# 7. RAG chain builder
# ═══════════════════════════════════════════════════════════════


def format_docs(docs: list[Document]) -> str:
    print(f"\n  >>> Retrieved {len(docs)} chunks")
    for i, d in enumerate(docs):
        page = d.metadata.get("page", "?")
        score = d.metadata.get("relevance_score", "?")
        print(f"    [{i}] page={page} score={score} | {d.page_content[:120]}...")
    return "\n\n---\n\n".join(
        f"[Trang {doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in docs
    )


def _debug_chain_step(inputs: dict) -> dict:
    question = inputs.get("question", "")
    context = inputs.get("context", "")
    print(f"\n>>> QUESTION: {question}")
    print(f">>> CONTEXT: {context[:500]}...")
    return inputs


def build_rag_chain(
    retriever: Runnable,
    system_prompt: str = SYSTEM_PROMPT,
) -> Runnable:
    """Build RAG chain với retriever injectable.

    Args:
        retriever: Bất kỳ Runnable[str | dict, list[Document]] nào
                   (VectorStoreRetriever, EnsembleRetriever, HybridRerankRetriever, ...)
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.0,
        max_tokens=512,
        google_api_key=settings.GEMINI_API_KEY,
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "NGỮ CẢNH:\n{context}\n\nCâu hỏi: {question}\n\n Trả lời:"),
    ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RunnableLambda(_debug_chain_step)
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return chain


# ═══════════════════════════════════════════════════════════════
# 8. Evaluation metrics
# ═══════════════════════════════════════════════════════════════


def normalize_text(text: str) -> str:
    return text.lower().strip()


def calc_keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    answer_lower = normalize_text(answer)
    matches = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return matches / len(expected_keywords)


def calc_context_recall(retrieved_docs: list[Document], expected_pages: list[int]) -> bool:
    if not expected_pages:
        return True
    retrieved_pages = set()
    for d in retrieved_docs:
        p = d.metadata.get("page")
        if p is not None:
            try:
                retrieved_pages.add(int(p) + 1)
            except (ValueError, TypeError):
                retrieved_pages.add(p)
    return any(ep in retrieved_pages for ep in expected_pages)


# ═══════════════════════════════════════════════════════════════
# 9. Evaluation runner
# ═══════════════════════════════════════════════════════════════


def run_evaluation(
    variant_name: str,
    retriever: Runnable,
    chain: Runnable,
    test_cases: list[TestCase],
) -> VariantResult:
    """Chạy evaluation trên 1 variant, trả về VariantResult."""
    details: list[CaseResult] = []
    total_cr = 0
    total_kw = 0.0
    cr_count = 0
    kw_count = 0
    total_latency = 0.0

    for tc in test_cases:
        t0 = time.time()
        retrieved_docs = retriever.invoke(tc.question)
        answer = chain.invoke(tc.question)
        latency = time.time() - t0
        total_latency += latency

        cr = calc_context_recall(retrieved_docs, tc.expected_source_pages)
        kw = calc_keyword_coverage(answer, tc.expected_answer_contains)

        retrieved_pages = []
        for d in retrieved_docs:
            p = d.metadata.get("page")
            if p is not None:
                try:
                    retrieved_pages.append(int(p) + 1)
                except (ValueError, TypeError):
                    retrieved_pages.append(p)
            else:
                retrieved_pages.append(None)

        keywords_found = [kw for kw in tc.expected_answer_contains if kw.lower() in normalize_text(answer)]

        if tc.expected_source_pages:
            total_cr += int(cr)
            cr_count += 1
        if tc.expected_answer_contains:
            total_kw += kw
            kw_count += 1

        details.append(CaseResult(
            id=tc.id,
            category=tc.category,
            question=tc.question,
            answer_preview=answer,
            context_recall=cr,
            keyword_coverage=round(kw, 4),
            latency_total=round(latency, 3),
            retrieved_pages=retrieved_pages,
            expected_pages=tc.expected_source_pages,
            expected_keywords=tc.expected_answer_contains,
            keywords_found=keywords_found,
        ))

        status = "OK" if (cr if tc.expected_source_pages else True) else "NG"
        print(f"  #{tc.id:2d} [{status}] {tc.category:14s} | CR={cr} KW={kw:.2f} | pages={retrieved_pages} | {latency:.1f}s")

    return VariantResult(
        variant=variant_name,
        num_cases=len(test_cases),
        context_recall=round(total_cr / cr_count, 4) if cr_count > 0 else None,
        keyword_coverage=round(total_kw / kw_count, 4) if kw_count > 0 else None,
        avg_latency=round(total_latency / len(test_cases), 3) if test_cases else None,
        details=details,
    )


# ═══════════════════════════════════════════════════════════════
# 10. Comparison & reporting
# ═══════════════════════════════════════════════════════════════


def _print_comparison_table(results: list[VariantResult]):
    header = f"{'Variant':20s} {'Context Rec':>12s} {'Keyword Cov':>12s} {'Avg Latency':>12s}"
    sep = "─" * 56
    print(f"\n{'=' * 56}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 56}")
    print(header)
    print(sep)
    for r in results:
        cr = f"{r.context_recall:.2%}" if r.context_recall is not None else "N/A"
        kw = f"{r.keyword_coverage:.2%}" if r.keyword_coverage is not None else "N/A"
        lat = f"{r.avg_latency:.2f}s" if r.avg_latency is not None else "N/A"
        print(f"{r.variant:20s} {cr:>12s} {kw:>12s} {lat:>12s}")
    print(f"{'=' * 56}\n")


def save_report(results: list[VariantResult], json_path: str):
    data = {}
    for r in results:
        data[r.variant] = r.model_dump()
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    txt_path = json_path.replace(".json", ".txt")
    lines = []
    for r in results:
        lines.append(f"{'=' * 60}")
        lines.append(f"VARIANT: {r.variant}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Total cases : {r.num_cases}")
        lines.append(f"  Context Rec : {r.context_recall}")
        lines.append(f"  Keyword Cov : {r.keyword_coverage}")
        lines.append(f"  Avg Latency : {r.avg_latency}s")
        lines.append(f"{'=' * 60}")
        lines.append("")
        for d in r.details:
            status = "OK" if (d.context_recall if d.expected_pages else True) else "NG"
            lines.append(f"  #{d.id:2d}  [{status}]  {d.category:14s}  CR={d.context_recall}  KW={d.keyword_coverage:.2f}  {d.latency_total:.1f}s")
            lines.append(f"       pages => {d.retrieved_pages}  (expected {d.expected_pages})")
            lines.append(f"       Q: {d.question}")
            lines.append(f"       A: {d.answer_preview[:120]}...")
            lines.append("")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  => Saved {json_path} and {txt_path}")


def run_comparison():
    """Build tất cả variant retriever → eval → print bảng so sánh."""
    print(f"\n{'=' * 60}")
    print("PREPARING DATA FOR COMPARISON")
    print(f"{'=' * 60}")

    print("[1/4] Loading PDF...")
    docs = load_pdf()
    print(f"  => {len(docs)} pages")

    print("[2/4] Splitting documents (chunk 512/128)...")
    chunks_512 = split_documents(docs, chunk_size=512, chunk_overlap=128)
    print(f"  => {len(chunks_512)} chunks")

    print("[3/4] Loading/creating vectorstores...")
    vsm_512 = VectorStoreManager(persist_directory=CHROMA_DIR, collection_name="chunk512va128")
    if Path(CHROMA_DIR).exists():
        vectorstore_512 = vsm_512.load()
    else:
        vectorstore_512 = vsm_512.create(chunks_512)

    variants: list[tuple[str, Runnable]] = [
        ("vector_only", build_vector_retriever(vectorstore_512, k=RETRIEVER_K)),
        ("ensemble_hybrid", build_ensemble_hybrid_retriever(vectorstore_512, chunks_512, k_vector=5, k_bm25=10)),
        ("reranked_hybrid", build_reranked_hybrid_retriever(vectorstore_512, chunks_512, k_bm25=15, k_vector=10, top_k=4)),
    ]

    results: list[VariantResult] = []
    for variant_name, retriever in variants:
        print(f"\n{'─' * 60}")
        print(f"EVALUATING: {variant_name}")
        print(f"{'─' * 60}")
        chain = build_rag_chain(retriever)
        result = run_evaluation(variant_name, retriever, chain, TEST_CASES)
        results.append(result)
        print(f"\n  >>> {variant_name}: CR={result.context_recall}  KW={result.keyword_coverage}  Lat={result.avg_latency}s")
        # rate-limit guard giữa các variant
        if variant_name != variants[-1][0]:
            print("  Waiting 5s before next variant...")
            time.sleep(5)

    _print_comparison_table(results)
    save_report(results, "comparison_report.json")
    print("Comparison complete.")


# ═══════════════════════════════════════════════════════════════
# 11. Main entry point
# ═══════════════════════════════════════════════════════════════


def test_single_chain():
    """Test nhanh 1 chain với reranked hybrid."""
    print("Loading documents...")
    docs = load_pdf()
    print(f"  => {len(docs)} pages")
    vsm = VectorStoreManager(persist_directory=CHROMA_DIR, collection_name="chunk1000va200")
    try:
        vectorstore = vsm.load()
    except Exception:
        print("Collection chunk1000va200 not found, falling back to chunk512va128...")
        vsm = VectorStoreManager(persist_directory=CHROMA_DIR, collection_name="chunk512va128")
        vectorstore = vsm.load()
    retriever = build_reranked_hybrid_retriever(vectorstore, docs, k_bm25=15, k_vector=10, top_k=4)
    chain = build_rag_chain(retriever)

    queries = [
        "Hãy hướng dẫn kết nối Wi-Fi cho điện thoại Samsung.",
        "Điện thoại Samsung của tôi bị treo logo, làm thế nào để khắc phục?",
        "Giá bán của sản phẩm này là bao nhiêu?",
    ]
    for q in queries:
        print(f"\n{'=' * 60}")
        print(f"Câu hỏi: {q}")
        print(f"Trả lời: {chain.invoke(q)}")


def main():
    import sys
    # if "--compare" in sys.argv:
    #     run_comparison()
    # elif "--test" in sys.argv:
    #     test_single_chain()
    # else:
    #     print("Usage: python index3.py [--compare | --test]")
    #     print("  --compare : run comparison of all retriever variants")
    #     print("  --test    : run a single test with reranked hybrid")
    #     print("\nRunning default: single test")
    test_single_chain()


if __name__ == "__main__":
    main()
