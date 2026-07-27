from pathlib import Path
import shutil
import time
from pydantic_settings import BaseSettings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from google.genai import types
from google import genai

class Settings(BaseSettings):
    GEMINI_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    model_config = {"env_file": ".env", "extra": "ignore"}

settings = Settings()

PDF_PATH = "documents/huongdansudungSamSung.pdf"
CHROMA_DIR = "./chroma_db_v2"
EMBED_BATCH_SIZE = 10
EMBED_BATCH_SLEEP = 10
RETRIEVE_K = 4
LOCAL_MODEL = "intfloat/multilingual-e5-large"

SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia hỗ trợ kỹ thuật và chăm sóc khách hàng của sản phẩm Samsung smartphone. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng bằng cách DỰA TRÊN NGỮ CẢNH được cung cấp từ tài liệu. Hãy tuân thủ nghiêm ngặt các yêu cầu sau:
1. Chỉ trả lời dựa trên thông tin trong đoạn trích. không suy đoán hoặc thêm thông tin bên ngoài.
2. Nếu đoạn trích không đủ thông tin để trả lời, hãy nói "Tài liệu không đề cập đến vấn đề này" và bỏ qua phần trích dẫn."
3. Trả lời theo nội dung trích dẫn, cùng với toàn bộ đoạn trích (nếu có), để người dùng có thể tham khảo. Nếu không có trích dẫn, chỉ cần trả lời ngắn gọn.
4. Trả lời bằng tiếng Việt. Thân thiện và dễ hiểu, không sử dụng từ ngữ chuyên ngành quá khó hiểu."""


# ─── Shared Utilities ───────────────────────────────────────────

def load_pdf(file_path: str) -> list:
    loader = PyPDFLoader(file_path)
    return loader.load()

def split_documents(docs: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", ";", ",", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        print(f"========= Chunk {i}: page={chunk.metadata.get('page', '?')} | {chunk.page_content[:80]}... ========")
    return chunks

def load_vectorstore(persist_dir: str = CHROMA_DIR, collection_name: str = "so_tay_samsung_genai", embedding_function=None) -> Chroma:
    if not Path(persist_dir).exists():
        raise ValueError(f"Thư mục {persist_dir} không tồn tại. Vui lòng tạo vectorstore trước.")
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_dir,
    )



# SECTION 1 — INDEX DÙNG LANGCHAIN  (Gemini Embeddings)


def create_embeddings_v1() -> GoogleGenerativeAIEmbeddings:
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY chưa được cấu hình trong .env")
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        api_key=settings.GEMINI_API_KEY,
    )

def create_vectorstore_v1(chunks: list, persist_dir: str = CHROMA_DIR) -> Chroma:
    if Path(persist_dir).exists():
        shutil.rmtree(persist_dir)

    emb = create_embeddings_v1()
    total = len(chunks)

    store = Chroma(
        collection_name="so_tay_samsung_genai",
        embedding_function=emb,
        persist_directory=persist_dir,
    )

    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        store.add_documents(batch)
        print(f"  → Batch {i//EMBED_BATCH_SIZE + 1}/{(total-1)//EMBED_BATCH_SIZE + 1}: {len(batch)} chunks")
        if i + EMBED_BATCH_SIZE < total:
            time.sleep(EMBED_BATCH_SLEEP)

    return store

def pipeline_indexing_langchain(pdf_path: str = PDF_PATH, chunk_size: int = 1000, chunk_overlap: int = 200):
    print("[1/4] đang load PDF....")
    docs = load_pdf(pdf_path)
    print(f"  → Loaded {len(docs)} pages")

    print("[2/4] đang split documents....")
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"  → Split into {len(chunks)} chunks")

    print("[3/4] đang tạo embeddings (Gemini) và lưu vào vectorstore....")
    store = create_vectorstore_v1(chunks)
    print(f"  → Đã lưu vectorstore tại {CHROMA_DIR}")

    return store



# SECTION 2 — QUERY & CHAT DÙNG LANGCHAIN  (LangChain LCEL)


def format_docs(docs: list) -> str:
    return "\n\n---\n\n".join(
        f"[Trang {d.metadata.get('page', '?')}] {d.page_content}"
        for d in docs
    )

def build_rag_prompt_with_langchain(query: str, results) -> str:
    context = "\n\n---\n\n".join(
        f"[Chunk {r.metadata.get('index', '?')}] {r.page_content}" for r in results
    )
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", f"NGỮ CẢNH\n{context}\n\nCâu hỏi: {query}\n\nTrả lời:"),
    ])
    return prompt_template.format()

def query_rag_chain_langchain(query: str, k: int = RETRIEVE_K) -> str:
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.0,
        google_api_key=settings.GEMINI_API_KEY,
    )

    store = load_vectorstore(
        persist_dir=CHROMA_DIR,
        collection_name="so_tay_samsung_genai",
        embedding_function=create_embeddings_v1(),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "NGỮ CẢNH:\n{context}\n\nCâu hỏi: {question}\n\nTrả lời:"),
    ])

    retriever = store.as_retriever(search_kwargs={"k": k})

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({"question": query})



# SECTION 3 — INDEX DÙNG LOCAL MODEL  (sentence-transformers)


def create_embeddings_v2() -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings(
        model_name=LOCAL_MODEL,
        model_kwargs={"device": "cpu"},
        show_progress=True,
    )

def create_vectorstore_v2(chunks: list, persist_dir: str = CHROMA_DIR) -> Chroma:
    if Path(persist_dir).exists():
        shutil.rmtree(persist_dir)

    emb = create_embeddings_v2()
    store = Chroma(
        collection_name="so_tay_samsung_local",
        embedding_function=emb,
        persist_directory=persist_dir,
    )
    store.add_documents(chunks)
    return store

def pipeline_indexing_local(pdf_path: str = PDF_PATH, chunk_size: int = 1000, chunk_overlap: int = 200):
    print("[1/4] đang load PDF....")
    docs = load_pdf(pdf_path)
    print(f"  → Loaded {len(docs)} pages")

    print("[2/4] đang split documents....")
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"  → Split into {len(chunks)} chunks")

    print("[3/4] đang tạo embeddings (local model) và lưu vào vectorstore....")
    store = create_vectorstore_v2(chunks)
    print(f"  → Đã lưu vectorstore tại {CHROMA_DIR}")

    return store



# SECTION 4 — CHAT DÙNG LOCAL MODEL  (genai.Client trực tiếp)

def setup_client() -> genai.Client:
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY chưa được cấu hình trong .env")
    return genai.Client(api_key=settings.GEMINI_API_KEY)

def build_rag_prompt(query: str, context: str) -> list[types.Content]:
   return [
        types.Content(
            role="user",
            parts=[types.Part(text=f"NGỮ CẢNH:\n{context}\n\nCâu hỏi: {query}\n\nTrả lời:")]
        )
    ]

def ask_llm(query: str, store: Chroma, k: int = RETRIEVE_K) -> str:
    client = setup_client()
    docs = store.similarity_search(query, k)

    context = "\n\n---\n\n".join(
        f"[Trang {d.metadata.get('page', '?')}] {d.page_content}"
        for d in docs
    )

    prompt_contents = build_rag_prompt(query, context)
    resp = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt_contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
        ),
    )
    return resp.text

def query_rag_chain_local(query: str, k: int = RETRIEVE_K) -> str:
    store = load_vectorstore(
        persist_dir=CHROMA_DIR,
        collection_name="so_tay_samsung_local",
        embedding_function=create_embeddings_v2(),
    )
    return ask_llm(query, store, k)



def main():
    # Chọn pipeline indexing: LangChain hoặc Local
    store = pipeline_indexing_langchain()
    # store = pipeline_indexing_local()

    # Thử truy vấn
    query = "1 cộng 1 bằng mấy?"
    # answer = query_rag_chain_local(query)
    answer = query_rag_chain_langchain(query)
    print(f"\nCâu hỏi: {query}\nTrả lời: {answer}")
    
    
if __name__ == "__main__":
    main()
