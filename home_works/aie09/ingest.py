from pydantic_settings import BaseSettings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters \
import RecursiveCharacterTextSplitter
from google.genai import types
from google import genai
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class Settings(BaseSettings):
    GEMINI_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    model_config = {"env_file": ".env", "extra": "ignore"}
    
PDF_PATH = "documents/huongdansudungSunhouse.pdf"

settings = Settings()

def load_pdf(file_path: str) -> list:
    """function load pdf file và trả về list các page"""
    loader = PyPDFLoader(file_path)
    return loader.load()

def split_documents(docs: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """function split document thành các chunk nhỏ hơn"""
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", ".", "?", "!", ";", ",", " ", ""], # custom lại separators
    add_start_index=True)
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        print(f"========= Chunk {i}:  Chunk page: {chunk.metadata.get('page', 'Unknown')} {chunk.page_content} ========")
    return chunks

emb = GoogleGenerativeAIEmbeddings(
model="gemini-embedding-001",)
store = Chroma(
collection_name="so_tay",
embedding_function=emb,
persist_directory="./chroma_db")
store.add_documents(chunks)


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