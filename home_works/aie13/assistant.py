## Trợ lý agentic RAG
# Tool 1) calculator           : tính biểu thức số học AN TOÀN (AST, không eval thô).
# Tool 2) search_docs(query)   : tra cứu vectorstore đã persist từ ingest.py
#                                (BM25 + semantic vector + CrossEncoder rerank).
#
# Yêu cầu hành vi của agent (sẽ xử lý bằng system prompt khi dựng create_agent):
# a) greeting (hello, how are you, good morning,...) -> trả lời tự nhiên, KHÔNG gọi tool.
# b) câu hỏi về nội dung tài liệu -> gọi search_docs; nếu kho không có ->
#    trả lời "Thông tin này không có trong tài liệu." (tuyệt đối không bịa).
# c) câu hỏi toán học -> gọi calculator để tính và trả kết quả.

# ===================== Tool 1: calculator =====================
import ast
import operator
from langchain.tools import tool

ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node):
    """Đệ quy tính một nút của cây cú pháp (AST). Chỉ cho phép phép toán trong ALLOWED_OPS."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op = type(node.op)
        if op not in ALLOWED_OPS:
            raise ValueError(f"Unsupported operator: {op.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if (op is ast.Div or op is ast.FloorDiv) and right == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return ALLOWED_OPS[op](left, right)
    if isinstance(node, ast.UnaryOp):
        op = type(node.op)
        if op not in ALLOWED_OPS:
            raise ValueError(f"Unsupported operator: {op.__name__}")
        return ALLOWED_OPS[op](_eval_node(node.operand))
    raise ValueError(f"Unsupported syntax: {type(node).__name__}")


def _format_number(value) -> str:
    """Bỏ đuôi .0 của số nguyên float, ví dụ 204.6 (giữ nguyên), 12.0 -> 12."""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.10g}"
    return str(value)


@tool
def calculator(expr: str) -> str:
    """Tính một biểu thức số học an toàn, ví dụ "102.3 * 2" hoặc "100 - 3 * 2".
    KHÔNG dùng eval() nguyên văn (nguy hiểm), mà parse sang AST rồi chỉ cho phép
    các phép toán trong ALLOWED_OPS. Trả về chuỗi kết quả, hoặc chuỗi bắt đầu bằng "Error:"."""
    try:
        tree = ast.parse(expr, mode="eval")
        return _format_number(_eval_node(tree.body))
    except ZeroDivisionError as e:
        return f"Error: {e}"
    except (SyntaxError, ValueError) as e:
        return f"Error: invalid expression '{expr}' ({e})"


# ===================== Tool 2: search_docs =====================
CHROMA_DIR = "./home_works/aie13/chroma_db_v1"
COLLECTION_NAME = "chunk800va150"
RETRIEVER_TOP_K = 4

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from sentence_transformers import CrossEncoder


def _create_embeddings_local() -> HuggingFaceEmbeddings:
    """hàm tạo embeddings model local với intfloat/multilingual-e5-large"""
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cpu"},
    )


class VectorStoreManager:
    """Mở Chroma đã persist từ ingest.py. KHÔNG embed lại, không load PDF."""

    def __init__(self, persist_directory: str = CHROMA_DIR, collection_name: str = COLLECTION_NAME,
                 create_embeddings=_create_embeddings_local):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        # lúc query vẫn cần embedding_function để embed câu hỏi, dùng đúng model đã index
        self.embeddings = create_embeddings()
        self.vectorstore: Chroma | None = None

    def load(self) -> Chroma:
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            collection_metadata={"hnsw:space": "cosine"},
        )
        print(f"  => Loaded vectorstore from {self.persist_directory}/{self.collection_name}")
        return self.vectorstore


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

    # vì sao lại phải deduplicate docs? vì BM25 và vector retriever có thể trả về các docs trùng nhau (với việc ta sử dụng các docs chunk đã split từ cùng 1 PDF),
    # nên cần deduplicate trước khi rerank để tránh việc rerank các chunk trùng nhau.
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
        # vì sao input là str | dict? vì retriever.invoke có thể nhận input là str hoặc dict,
        # nếu là dict thì lấy key "question" hoặc "input" để làm query, nếu là str thì dùng trực tiếp
        query = input.get("question") or input.get("input") or str(input) if isinstance(input, dict) else str(input)

        bm25_docs = self.bm25.invoke(query, config=config)
        vector_docs = self.vector_ret.invoke(query, config=config)

        print(f"\n  [DEBUG] hybrid reranker cho query='{query}'")
        print(f"  [DEBUG] BM25 retrieved {len(bm25_docs)} docs (k={self.bm25.k})")
        
        print(f"  [DEBUG] Vector retrieved {len(vector_docs)} docs (k={self.vector_ret.search_kwargs.get('k')})")

        # kết hợp các docs từ BM25 và vector retriever
        candidate_docs = self.deduplicate_docs(bm25_docs + vector_docs)
        print(f"  [DEBUG] After dedup: {len(bm25_docs) + len(vector_docs)} candidates -> {len(candidate_docs)}")
        # rerank các docs đã deduplicate
        reranked_docs = self.reranker.rerank(query, candidate_docs)
        return reranked_docs


def create_hybrid_rerank_retriever(
    vectorstore: Chroma,
    documents: list[Document],
    k_bm25: int = 15,
    k_vector: int = 10,
    top_k: int = 5,
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


def _chunks_from_vectorstore(vectorstore: Chroma) -> list[Document]:
    """Đọc toàn bộ chunk ĐÃ INDEX trong Chroma -> nguồn cho BM25.

    Khóa vấn đề "load PDF + split lại mỗi lần": chunk lấy ra từ chính kho đã persist,
    không mở lại PDF, không split lại, và luôn khớp dữ liệu trong collection."""
    data = vectorstore.get(limit=10**7)
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(data["documents"], data["metadatas"])
    ]


_retriever_cache: HybridRerankRetriever | None = None


def _get_hybrid_retriever() -> HybridRerankRetriever:
    """Lazy-singleton: mở kho + build retriever đúng 1 lần, các lần sau dùng lại cache."""
    global _retriever_cache
    if _retriever_cache is None:
        vectorstore = VectorStoreManager().load()
        chunks = _chunks_from_vectorstore(vectorstore)
        print(f"  => Đọc {len(chunks)} chunks từ Chroma '{COLLECTION_NAME}' để dựng BM25 + rerank")
        _retriever_cache = create_hybrid_rerank_retriever(
            vectorstore=vectorstore,
            documents=chunks,
            k_bm25=15,
            k_vector=10,
            top_k=RETRIEVER_TOP_K,
        )
    return _retriever_cache


def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('source', '?')} - Trang {doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in docs
    )


@tool
def search_docs(query: str) -> str:
    """Tìm kiếm thông tin trong tài liệu nội bộ (có thể có nhiều tài liệu).
    Trả về các đoạn liên quan nhất kèm [Tên tài liệu - Trang]. Chỉ dùng khi user hỏi về nội dung
    CÓ THỂ có trong tài liệu; nếu không liên quan sẽ không trả về gì."""
    retrieved = _get_hybrid_retriever().invoke(query)
    print(f"\n  >>> Retrieved {len(retrieved)} chunks cho query: {query}")
    for i, d in enumerate(retrieved):
        page = d.metadata.get("page", "?")
        score = d.metadata.get("relevance_score", "?")
        print(f"    [{i}] page={page} score={score} | {d.page_content[:120]}...")
    return format_docs(retrieved)


## ========== phần 2, thiết kế agentic RAG agent ==========
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from uuid import uuid4

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.0,
    max_output_tokens=1024,
)

TOOLS = [calculator, search_docs]

SYSTEM_PROMPT = """Bạn là trợ lý thực hiện tra cứu thông tin trong các tài liệu nội bộ và trả lời câu hỏi.
    Bạn có thể thực hiện các hành vi sau:
    1) các câu hỏi xã giao (hello, how are you, good morning,...) -> trả lời tự nhiên, KHÔNG gọi tool.
    2) câu hỏi về nội dung tài liệu -> gọi search_docs(query); nếu kho không có ->
    trả lời "Thông tin này không có trong tài liệu." (tuyệt đối không bịa).
    3) câu hỏi toán học -> gọi calculator(expr) để tính và trả kết quả.
    Lưu ý: 
    - KHÔNG bịa thông tin, KHÔNG trả lời nếu không chắc chắn, KHÔNG trả lời ngoài phạm vi tài liệu.
    - Sẽ có các câu hỏi cần context từ các câu hỏi trước (multi-turn), bạn cần kiểm tra context trước khi trả lời.
    - Khi trả lời, nếu có thông tin từ tài liệu, hãy dẫn nguồn [Tên tài liệu - Trang X].
    """

def _content_to_text(content) -> str:
    """Gemini (langchain_google_genai 4.x) trả content dạng list block:
    [{'type':'text','text':...}, ...]. Hàm này rút phần text thuần để in đẹp."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif "text" in block:
                    parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return str(content)

def _last_text(messages) -> str:
    """Lấy nội dung text của AIMessage cuối cùng trong list messages."""
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if content:
            return _content_to_text(content)
    return str(messages[-1])

def _run_chat_cli():
    """Vòng chat CLI với memory theo thread.

    Lệnh:
      /new  (hoặc new) -> tạo thread_id mới => quên toàn bộ lịch sử phiên cũ
      /quit (hoặc exit) -> thoát chương trình
    """
    memory = InMemorySaver()
    thread_id = f"thread_{uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    agent = create_agent(
        tools=TOOLS,
        model=llm,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )

    # warm-up retriever trước vòng lặp để lần hỏi đầu không bị tê liệt
    _get_hybrid_retriever()
    print(f"AIE 13 Agentic RAG (gõ '/new' để bắt đầu phiên mới, '/quit' để thoát).")
    print(f"Phiên hiện tại: {thread_id}")

    while True:
        query = input("\nBạn: ").strip()
        if not query:
            continue
        if query.lower() in ("/quit", "exit"):
            print("Tạm biệt!")
            break
        if query.lower() in ("/new", "new"):
            # đổi sang thread mới: cùng InMemorySaver nhưng config mới => mất "kí ức" phiên cũ
            thread_id = f"thread_{uuid4().hex[:8]}"
            config = {"configurable": {"thread_id": thread_id}}
            print(f"Đã bắt đầu phiên mới: {thread_id}")
            continue

        print(f"\n=== User Query ===\n{query}")
        print("\n=== Agent Processing ===\n")
        final_text = None
        for step in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="updates",
            config=config,
        ):
            for node_name, state in step.items():
                msg = state.get("messages", [])[-1]
                if getattr(msg, "tool_calls", None):
                    print(f"  - node '{node_name}' -> gọi tool: {msg.tool_calls}")
                else:
                    text = _content_to_text(getattr(msg, "content", None))
                    print(f"  - node '{node_name}' -> text: {text}")
                    # message cuối cùng do node 'agent' sinh ra chính là câu trả lời
                    if node_name == "model":
                        final_text = text

        print(f"\nTrợ lý: {final_text or '(không có phản hồi)'}\n")


if __name__ == "__main__":
    # _test_search()
    _run_chat_cli()