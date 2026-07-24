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


# ─────────────── INDEXING ───────────────

def load_pdf(file_path: str = PDF_PATH) -> list:
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_documents(docs: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
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


# ─────────────── QUERY ───────────────

def format_docs(docs: list) -> str:
    return "\n\n---\n\n".join(
        f"[Trang {doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in docs
    )

def seteup_genai_client() -> genai.Client:
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY chưa được cấu hình trong .env")
    return genai.Client(api_key=settings.GEMINI_API_KEY)

client = seteup_genai_client()

def build_rag_chain(vectorstore: Chroma, k: int = RETRIEVER_K):
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        google_api_key=settings.GEMINI_API_KEY,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "NGỮ CẢNH:\n{context}\n\nCâu hỏi: {question}\n\nTrả lời:"),
    ])

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask(chain, question: str) -> str:
    return chain.invoke(question)


# ─────────────── CHAT (KHÔNG RE-INDEX) ───────────────

_chain = None

def _ensure_chain(k: int = RETRIEVER_K):
    global _chain
    if _chain is None:
        vstore = load_vectorstore()
        _chain = build_rag_chain(vstore, k=k)
    return _chain

def chat(question: str, k: int = RETRIEVER_K) -> str:
    chain = _ensure_chain(k=k)
    return ask(chain, question)

def reset_chain():
    global _chain
    _chain = None


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


# ─────────────── PIPELINE ───────────────

def index_pipeline(pdf_path: str = PDF_PATH):
    print("[1/4] Đang load PDF...")
    docs = load_pdf(pdf_path)
    print(f"  → {len(docs)} trang")

    print("[2/4] Đang split documents...")
    chunks = split_documents(docs)
    print(f"  → {len(chunks)} chunks")

    print("[3/4] Đang tạo embeddings và lưu Chroma (rate-limited)...")
    vectorstore = create_vectorstore(chunks)
    print(f"  → Đã lưu tại {CHROMA_DIR}")

    print("[4/4] Xây dựng RAG chain...")
    chain = build_rag_chain(vectorstore)

    print("\n>>> Test 3 câu hỏi:")
    test_queries(chain)

    return chain


if __name__ == "__main__":
    if not settings.GEMINI_API_KEY:
        print("Lỗi: GEMINI_API_KEY chưa được cấu hình trong .env")
    else:
        # index_pipeline()
        test_queries()
