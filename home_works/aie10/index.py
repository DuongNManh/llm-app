import time
import shutil
from pathlib import Path
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


# ─────────────── INDEXING ───────────────

def load_pdf(file_path: str = PDF_PATH) -> list:
    """Load PDF và trả về danh sách các trang dưới dạng Document."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_documents(docs: list, chunk_size: int = 512, chunk_overlap: int = 128) -> list:
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
    if Path(persist_dir).exists():
        shutil.rmtree(persist_dir)

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
    embeddings = create_embeddings()
    return Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


# ─────────────── RETRIEVAL ───────────────

def format_docs(docs: list) -> str:
    return "\n\n---\n\n".join(
        f"[Trang {doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in docs
    )


def build_vector_retriever(vectorstore, k: int = RETRIEVER_K):
    """Pure vector search retriever (baseline)."""
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_hybrid_retriever(vectorstore, documents, k: int = HYBRID_K):
    """Hybrid retriever: BM25 (keyword) + Vector (semantic).

    Kỹ thuật cải tiến chính:
    - BM25 bắt chính xác từ khóa (model number, tên lỗi)
    - Vector hiểu ngữ nghĩa (synonym, paraphrase)
    - Kết hợp bằng weighted fusion (0.5/0.5)
    - k=15 lấy rộng làm ứng viên
    """
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
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
    chunks = split_documents(docs)
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
                _chunks = split_documents(docs)
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
        "Hãy hướng dẫn kết nối Wifi cho điện thoại Samsung.",
        "Điện thoại Samsung của tôi bị treo logo, làm thế nào để khắc phục?",
        "Giá bán của sản phẩm này là bao nhiêu?",
    ]
    for q in queries:
        print(f"\n{'='*60}")
        print(f"Câu hỏi: {q}")
        print(f"Trả lời: {ask(chain, q)}")
    print()


def ask(chain, question: str) -> str:
    return chain.invoke(question)


if __name__ == "__main__":
    if not settings.GEMINI_API_KEY:
        print("Lỗi: GEMINI_API_KEY chưa được cấu hình trong .env")
    else:
        index_pipeline()
