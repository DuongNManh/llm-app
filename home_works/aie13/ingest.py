from pydantic_settings import BaseSettings
from langchain_community.document_loaders import PyPDFLoader

# setup env
class Settings(BaseSettings):
    GEMINI_API_KEY : str | None = None
    TAVILY_API_KEY : str | None = None
    OPENROUTER_API_KEY : str | None = None
    model_config = {"env_file": ".env", "extra": "ignore"}

settings = Settings()

PDF1_PATH = "documents/huongdansudungSamSung.pdf"
CHROMA_DIR = "./home_works/aie13/chroma_db_v1"
EMBED_BATCH_SIZE = 10
EMBED_BATCH_SLEEP = 10
RETRIEVER_K = 4
HYBRID_K = 15

## ========== load PDF và split thành chunk ==========

def load_pdf_documents(file_path) -> list:
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading PDF documents: {e}")
        return []

from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(docs: list, chunk_size: int = 800, chunk_overlap: int = 150) -> list:
    """split các trang doc thành các chunk và có overlap"""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "?", "!", ";", ",", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(docs)

from langchain_community.embeddings import HuggingFaceEmbeddings

def _create_embeddings_local() -> HuggingFaceEmbeddings:
    """hàm tạo embeddings model local với intfloat/multilingual-e5-large"""
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cpu"},
    )
    
## VectorStoreManager class để em quản lý tạo và load vectorstore

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class VectorStoreManager:
    """VectorStoreManager class để quản lý việc tạo và load vectorstore sử dụng Chroma."""
    def __init__(self, persist_directory: str = CHROMA_DIR, collection_name: str = "chunk800va120", create_embeddings=_create_embeddings_local):
        self.persist_directory = persist_directory # nơi lưu trữ vectorstore
        self.collection_name = collection_name # em đặt tên collection p.biệt các variant chunk size
        self.embeddings = create_embeddings() # nhét hàm embeddings vào, lúc tạo vector và lúc load vector đều dùng chung embeddings này
        self.vectorstore: Chroma | None = None # vectorstore của ứng dụng, lúc tạo thì là Chroma.from_documents, lúc load thì là Chroma(...)

    def create(self, documents: list[Document]) -> Chroma:
        self.vectorstore = Chroma.from_documents(
            documents=documents, # documents là list[Document] đã split từ PDF
            embedding=self.embeddings, # hàm embeddings, lúc tạo vector và load vector đều dùng chung embeddings này 
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self.vectorstore.persist()
        print(f"  => Đã tạo vectorstore: {len(documents)} docs tại {self.persist_directory}")
        return self.vectorstore

    def load(self) -> Chroma:
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            collection_metadata={"hnsw:space": "cosine"},
        )
        print(f"  => Đã load vectorstore từ {self.persist_directory}/{self.collection_name}")
        return self.vectorstore


## Các retriever variant

# 1. hybrid retriever reranker với BM25Retriever

from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from langchain_core.runnables import Runnable

_global_reranker = CrossEncoder("BAAI/bge-reranker-large", device="cpu", max_length=512)

# class này dùng để rerank lại các docs đã retrieve dựa trên score của CrossEncoder
class CrossEncoderReranker:
    """Nhận query và danh sách docs, trả về top-k docs đã rerank"""
    def __init__(self, model: CrossEncoder, top_k: int = 4, batch_size: int = 15):
        self.model = model
        self.top_k = top_k
        self.batch_size = batch_size

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        if len(docs) <= self.top_k:
            return docs
        
        # pair các query với các docs
        pairs = [(query, doc.page_content) for doc in docs]
        # tính score cho các cặp query-docs
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        # preview doc và score theo query
        print(f"\n  [DEBUG] Rerank {len(docs)} docs for query='{query}'")
        for i, (score, doc) in enumerate(zip(scores, docs)):
            print(f"    [{i+1}] score={score:.4f} | page={doc.metadata.get('page')} | {doc.page_content[:100]}...")

        # sắp xếp doc theo key là score giảm dần (ta có pair x[0,1] là (score, doc), nên x[0] là score)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        result = []
        # lấy top-k doc đã rerank trong list ranked
        for score, doc in ranked[:self.top_k]:
            # gắn thêm score vào metadata của doc để tiện cho việc hiển thị sau này
            doc.metadata["relevance_score"] = float(score)
            print(f"    [TOP] score={score:.4f} | page={doc.metadata.get('page')} | {doc.page_content[:100]}...")
            result.append(doc)
        return result

# class này dùng để kết hợp BM25 + Semantic -> dedup -> CrossEncoder rerank -> top-k
class HybridRerankRetriever(Runnable[str, list[Document]]):
    """BM25 + Semantic -> dedup -> CrossEncoder rerank -> top-k"""
    
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
    
    # vì sao lại phải deduplicate docs? vì BM25 và vector retriever có thể trả về các docs trùng nhau, nên cần deduplicate trước khi rerank
    # doc của vector store có khác doc của BM25 ko?
    # trả lời: ta đã split các doc trong PDF thành các chunk nhỏ, vector store lưu các chunk này, và BM25 tìm kiếm trên các chunk này, nên có thể trả về các chunk trùng nhau. Vì vậy, cần deduplicate trước khi rerank để tránh việc rerank các chunk trùng nhau.
    @staticmethod
    def deduplicate_docs(docs: list[Document]) -> list[Document]:
        seen = set()
        unique_docs = []
        for doc in docs:
            key = (
                doc.metadata.get("id")
                or (doc.metadata.get("source"), doc.metadata.get("page"), doc.metadata.get("start_index"))
            )
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
        return unique_docs

    def invoke(self, input: str | dict, config=None, **kwargs) -> list[Document]:
        # vì sao input là str | dict? vì retriever.invoke có thể nhận input là str hoặc dict, nếu là dict thì lấy key "question" hoặc "input" để làm query, nếu là str thì dùng trực tiếp
        query = input.get("question") or input.get("input") or str(input) if isinstance(input, dict) else str(input)
        
        bm25_docs = self.bm25.invoke(query, config=config)
        vector_docs = self.vector_ret.invoke(query, config=config)

        print(f"\n  [DEBUG] Hybrid Reranker Retriever cho query='{query}'")
        print(f"  [DEBUG] BM25 retrieved {len(bm25_docs)} docs (k={self.bm25.k})")
        print(f"  [DEBUG] Vector retrieved {len(vector_docs)} docs (k={self.vector_ret.search_kwargs.get('k')})")
        # kết hợp các docs từ BM25 và vector retriever
        candidate_docs = self.deduplicate_docs(bm25_docs + vector_docs)
        print(f"   [DEBUG] After dedup: {len(bm25_docs) + len(vector_docs)} candidates -> {len(candidate_docs)}")
        # rerank các docs đã deduplicate
        reranked_docs = self.reranker.rerank(query, candidate_docs)
        return reranked_docs

def create_hybrid_rerank_retriever(
    vectorstore: Chroma,
    documents: list[Document],
    k_bm25: int = 15,
    k_vector: int = 10,
    top_k = 5,
    reranker: CrossEncoderReranker | None = None,
    ) -> HybridRerankRetriever:
    """Tạo một HybridRerankRetriever với BM25 + Semantic -> dedup -> CrossEncoder rerank -> top-k"""
    # check var trước reranker, nếu reranker là None thì tạo một CrossEncoderReranker mới với top_k
    if reranker is None:
        reranker = CrossEncoderReranker(model=_global_reranker, top_k=top_k)
    return HybridRerankRetriever(
        documents=documents,
        vectorstore=vectorstore,
        reranker=reranker,
        k_bm25=k_bm25,
        k_vector=k_vector,
    )

def format_docs(docs: list[Document]) -> str:
    """format list[Document] trở thành string để hiển thị, mỗi doc là [Trang {page}, Document {source}] {content}"""
    return "\n\n---\n\n".join(
        f"[Trang {doc.metadata.get('page', '?')}, Document {doc.metadata.get('source', '?')}] {doc.page_content}"
        for doc in docs
    )

COLLECTION_NAME = "chunk800va150"

def _stable_chunk_id(chunk: Document) -> str:
    """Tạo id ổn định cho chunk từ (source, page, start_index) để chạy lại không nhân đôi."""
    return (
        f"{chunk.metadata.get('source')}#p{chunk.metadata.get('page')}"
        f"#start{chunk.metadata.get('start_index')}"
    )


def ingest(
    pdf_path: str = PDF1_PATH,
    persist_directory: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    chunk_size: int = 800,
    chunk_overlap: int = 150):
# ) -> Chroma:
    """Pipeline index: đọc PDF -> chunk -> gán id -> embed -> lưu Chroma.

    Idempotent: chỉ thêm những chunk chưa tồn tại trong collection (dựa trên id ổn định),
    nên chạy lại nhiều lần không bị nhân đôi dữ liệu. In ra số chunk đã index."""
    # bước 1: đọc PDF
    print("[ingest][1/3] Đang đọc PDF....")
    documents = load_pdf_documents(pdf_path)
    print(f"  -> Loaded {len(documents)} pages")

    # bước 2: split thành chunk
    print(f"[ingest][2/3] Đang split documents (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for chunk in chunks:
        chunk.metadata["id"] = _stable_chunk_id(chunk)
    print(f"  -> Đã split thành {len(chunks)} chunks")

    # bước 3: index vào Chroma, chỉ add chunk chưa có
    print("[ingest][3/3] Đang index vào Chroma....")
    manager = VectorStoreManager(persist_directory=persist_directory, collection_name=collection_name)
    vectorstore = manager.load()
    existing_ids = set(vectorstore.get(limit=10**7)["ids"])
    new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]
    if new_chunks:
        vectorstore.add_documents(new_chunks, ids=[c.metadata["id"] for c in new_chunks])
        print(f"  -> Đã thêm {len(new_chunks)} chunks (bỏ qua {len(chunks) - len(new_chunks)} đã có)")
    else:
        print(f"  -> Không có chunk mới, kho đã đủ {len(existing_ids)} chunks")

    total = len(existing_ids) + len(new_chunks)
    print(f"[ingest] HOÀN TẤT: {total} chunks trong collection '{collection_name}' tại {persist_directory}")
    return vectorstore


# ==== DEBUG test retrieval ====

# biến global để cache hybrid retriever, tránh việc tạo lại nhiều lần
_retriever_cache: HybridRerankRetriever | None = None

# hàm lấy các chunk từ vectorstore để làm nguồn cho BM25, ko mở lại pdf và split
def _chunks_from_vectorstore(vectorstore: Chroma) -> list[Document]:
    """Đọc toàn bộ chunk ĐÃ INDEX trong Chroma -> nguồn cho BM25.

    Khóa vấn đề "load PDF + split lại mỗi lần": chunk lấy ra từ chính kho đã persist,
    không mở lại PDF, không split lại, và luôn khớp dữ liệu trong collection."""
    data = vectorstore.get(limit=10**7)
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(data["documents"], data["metadatas"])
    ]

def _get_hybrid_retriever() -> HybridRerankRetriever:
    """Lazy singleton pattern: tạo hybrid retriever 1 lần, cache lại và dùng cho các lần sau."""
    # sử dụng keyword global để truy cập biến _retriever_cache bên ngoài hàm để thay đổi giá trị
    global _retriever_cache
    if _retriever_cache is None:
        print("[DEBUG] Load vectorstore và tạo hybrid retriever lần đầu...")
        # em load vectorstore có collection name là chunk800va150 và embedding function
        vectorstore = VectorStoreManager(persist_directory=CHROMA_DIR, collection_name=COLLECTION_NAME, create_embeddings=_create_embeddings_local).load()
        # em load các chunk từ vectorstore lên để cache lại
        chunks = _chunks_from_vectorstore(vectorstore)
        print(f"[DEBUG] => Đã đọc {len(chunks)} từ Chroma: {CHROMA_DIR}/{COLLECTION_NAME}")
        _retriever_cache = create_hybrid_rerank_retriever(
            vectorstore= vectorstore,
            documents= chunks,
            k_bm25=15,
            k_vector=10,
            top_k=4
        )
    return _retriever_cache

def search(query: str) -> str:
    """ Hàm search tool với input là query, trả về đoạn văn trong tài liệu liên quan nhất đến query theo dạng [Trang] [Document Name] [Content]. Sử dụng để tra cứu thông tin chính xác từ tài liệu nội bộ."""
    # tạo hybrid retriever có sẵn Chroma và list các docs chunk
    hybrid_retriever = _get_hybrid_retriever()
    # retrieve docs
    retrieved_docs = hybrid_retriever.invoke(query)
    # format docs thành str
    print(f"\n  >>> Retrieved {len(retrieved_docs)} chunks for query: {query}")
    for i, d in enumerate(retrieved_docs):
        page = d.metadata.get("page", "?")
        score = d.metadata.get("relevance_score", "?")
        print(f"    [{i}] page={page} score={score} | {d.page_content[:120]}...")
    return format_docs(retrieved_docs)


def _test_search():
    query = "Dung lượng của SM-A125F?"
    result = search(query)
    print(f"\n\n=== Search Result ===\n{result}")


## ============== Main ingest và test ====================
def main():
    ingest(pdf_path=PDF1_PATH, persist_directory=CHROMA_DIR, collection_name=COLLECTION_NAME, chunk_size=800, chunk_overlap=150)
    # _test_search()


if __name__ == "__main__":
    main()