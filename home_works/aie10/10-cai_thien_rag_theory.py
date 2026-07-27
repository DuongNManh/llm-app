### Cải thiện RAG qua chunk khéo, hybrid, reranking, đo lường

## RAG demo được != RAG tốt:
# Naive RAG dễ fail vì phần lớn nằm ở step retrieval, nếu retriever không tìm được ngữ cảnh tốt, thì LLM sẽ không có thông tin để trả lời câu hỏi.

# example: retrieve trả về đoạn không liên quan, LLM sẽ trả lời sai, hoặc trả lời mập mờ, hoặc trả lời "Tôi không biết". chất lượng đầu vào kém => chất lượng đầu ra kém.

# chất lượng RAG có thể cải thiện bằng cách:
# 1. chunk khéo: tách ngữ cảnh thành các chunk nhỏ, nhưng vẫn giữ nguyên ngữ cảnh, không tách giữa các câu, giữa các đoạn văn, giữa các ý. Có thể sử dụng RecursiveCharacterTextSplitter của langchain để tách ngữ cảnh.
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(docs: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """Split các trang thành các đoạn nhỏ hơn để tạo embeddings."""
    # RecursiveCharacterTextSplitter sẽ thử nhiều separator khác nhau, từ separator dài đến separator ngắn, để tách văn bản.
    # CharacterTextSplitter chỉ tách theo một separator duy nhất, nên có thể tách không tốt, ví dụ tách giữa 2 câu, hoặc giữa 2 đoạn văn.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ".", "?", "!", ";", ",", " ", ""],
    )
    return splitter.split_documents(docs)

# 2. hybrid: kết hợp nhiều vectorstore khác nhau, ví dụ: Chroma + FAISS, để tăng khả năng tìm kiếm ngữ cảnh.
