from pydantic_settings import BaseSettings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


PDF_PATH = "documents/huongdansudungSamSung.pdf"
CHROMA_DIR = "./chroma_db"
EMBED_BATCH_SIZE = 10
EMBED_BATCH_SLEEP = 10
RETRIEVER_K = 4

SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia hỗ trợ kỹ thuật và chăm sóc khách hàng của sản phẩm Samsung Smart Phone. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng bằng cách DỰA TRÊN NGỮ CẢNH được cung cấp từ tài liệu. Hãy tuân thủ nghiêm ngặt các yêu cầu sau:
1. Chỉ trả lời dựa trên thông tin trong đoạn trích. không suy đoán hoặc thêm thông tin bên ngoài.
2. Nếu đoạn trích không đủ thông tin để trả lời, hãy nói "Tài liệu không đề cập đến vấn đề này."
3. Trả lời theo nội dung trích dẫn, cùng với toàn bộ đoạn trích (nếu có) chứa câu trả lời.
4. Trả lời bằng tiếng Việt, thân thiện và dễ hiểu, không sử dụng từ ngữ chuyên ngành quá khó hiểu."""

class Settings(BaseSettings):
    """Cấu hình cho ứng dụng."""
    GEMINI_API_KEY: str | None = None
    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()


def create_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Tạo embeddings sử dụng Google Generative AI Embeddings với model "gemini-embedding-001"."""
    gemini_api_key = settings.GEMINI_API_KEY
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY chưa được cấu hình trong .env")
    
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        api_key=gemini_api_key,
    )


CHUNK1000va200 = "chunk1000va200"
CHUNK512va128 = "chunk512va128"

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

def format_docs(docs: list) -> str:
    """Format các đoạn trích từ tài liệu thành một chuỗi văn bản, bao gồm số trang và nội dung của từng đoạn. Trong đó, các đoạn trích được phân tách bằng chuỗi "\n\n---\n\n"."""
    # vì sao phải định dạng lại docs? vì retriever trả về docs là list of Document, nhưng LLM chỉ nhận input là str, nên cần format lại docs thành str
    return "\n\n---\n\n".join(
        f"[Trang {doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in docs
    )
    
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.vectorstores import VectorStoreRetriever

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
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return chain


def ask(chain, question: str) -> str:
    return chain.invoke(question)

def main():
    # load vectorstore từ thư mục lưu trữ Chroma
    vectorstore1 = load_vectorstore(per_dir=CHROMA_DIR, collection_name=CHUNK512va128)
    # build rag chain
    chain = build_rag_chain(vectorstore1)
    # hỏi câu hỏi mẫu
    # question = "Làm thế nào để reset điện thoại Samsung?"
    # print(f"Câu hỏi: {question}")
    # # chạy chain với câu hỏi và in ra kết quả
    # answer = ask(chain, question)
    # print(f"Trả lời: {answer}")

    # 3 trường hợp câu hỏi mẫu để test
    # 1. Câu hỏi có câu trả lời trong ngữ cảnh
    # 2. Câu hỏi mập mờ, có ý diễn đạt khác so với ngữ cảnh
    # 3. Câu hỏi không có câu trả lời trong ngữ cảnh
    queries = [
            "Hãy hướng dẫn kết nối Wifi cho điện thoại Samsung.",
            "Điện thoại của tôi bị treo logo, làm thế nào để khắc phục?",
            "Giá bán của sản phẩm này là bao nhiêu?",
        ]
    
    user_questions = input("Nhập câu hỏi của bạn (hoặc nhấn Enter để sử dụng câu hỏi mẫu): ")
    if user_questions.strip():
        queries = [user_questions.strip()]
    
    for q in queries:
        print(f"\n{'='*60}")
        print(f"Câu hỏi: {q}")
        print(f"Trả lời: {ask(chain, q)}")
        print()

if __name__ == "__main__":
    main()