import ast
import operator
import unicodedata

from typing import Annotated, TypedDict

from pydantic_settings import BaseSettings
from pathlib import Path

# LangChain / LangGraph
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages   # reducer: gộp lịch sử chat
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver  # checkpointer lưu memory


# ============================================================
# 0. CẤU HÌNH MÔI TRƯỜNG
# ============================================================
class Settings(BaseSettings):
    OPENROUTER_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    TAVILY_API_KEY: str | None = None
    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()

if not settings.GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

# Dùng ChatGoogleGenerativeAI (langchain_google_genai) chứ KHÔNG phải google.genai thô,
# vì LangGraph cần model implement giao diện BaseChatModel có bind_tools() (tool-calling).
MODEL = "gemini-3.1-flash-lite"

llm = ChatGoogleGenerativeAI(model=MODEL, api_key=settings.GEMINI_API_KEY, temperature=0.0)

# kho dữ kiện
KNOWLEDGE = {
    "dân số việt nam 2026": "102.3 triệu người (2026)",
    "số thiết bị java 2026": "3 tỷ thiết bị (2026)",
    "gdp việt nam 2024": "430 tỷ USD (2024)",
    "chiều cao đỉnh everest": "8848.86 mét",
    "dân số nhật bản 2025": "123.9 triệu người (2025)",
    "tốc độ ánh sáng": "299792458 mét/giây",
}


def _normalize(text: str) -> str:
    """Chuẩn hóa văn bản không dấu, chữ thường để so khớp câu hỏi với KNOWLEDGE."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower().strip()


# ============================================================
# 1. ĐỊNH NGHĨA 2 TOOL BẰNG @tool  (Yêu cầu số 2)
# ============================================================
# @tool biến hàm thường thành "tool" mà LLM hiểu được qua bind_tools().
# Phần DOCSTRING rất quan trọng: framework truyền docstring cho model
# để model quyết định chọn tool nào và gọi với đối số nào.

@tool
def search(query: str) -> str:
    """Tra cứu dữ kiện đã biết (dân số, GDP, chiều cao, tốc độ...). 
    Trả về dữ kiện khớp với query, hoặc thông báo không tìm thấy."""
    norm_q = _normalize(query)
    for key, value in KNOWLEDGE.items():
        norm_key = _normalize(key)
        if norm_q in norm_key or norm_key in norm_q:
            return f"{key}: {value}"
    return f"Không tìm thấy dữ kiện cho: {query}"

from langchain_community.tools.tavily_search import TavilySearchResults

@tool
def tanaly_tool():
    """Tool tìm kiếm thông tin tổng hợp trên internet về các sự kiện, tin tức mới nhất."""
    search_tool = TavilySearchResults(max_results=3, api_key=settings.TAVILY_API_KEY)
    return search_tool


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
    """Đệ quy tính một nút của cây cú pháp (AST). Tự bàn thêm ở calculator."""
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


TOOLS = [search, calculator]

# ============================================================
# 2. HỆ PROMPT (text thuần — create_agent tự nạp tools vào model)
# ============================================================
# Với create_agent, KHÔNG truyền {tools}/{input}/{agent_scratchpad}: framework tự
# bind tools vào model và tự chèn kết quả tool (ToolMessage) vào vòng lặp.
SYSTEM_PROMPT = (
    "Bạn là một agent tra cứu và tính toán."
    "Khi cần một dữ kiện (dân số, số liệu, tốc độ...) hãy dùng tool search(query) với query đúng như yêu cầu, không thêm hay bớt dữ kiện. "
    "Khi cần phép tính số học hãy dùng tool calculator(expr). "
    "Trả lời cuối cùng bằng tiếng Việt, ngắn gọn."
)


# ============================================================
# PHẦN 1 — ĐƯỜNG NHANH: create_agent
# ============================================================
def run_with_create_agent(question: str) -> list[str]:
    """Cách 1: để framework dựng toàn bộ graph (agent node + ToolNode + tools_condition gói sẵn).

    Vào:  invoke( {"messages":[{"role":"user","content":Q}]} )  -> dict có "messages".
    """
    # create_agent gói 3 thành phần bạn cần: agent node (bind_tools),
    # tools node đi kèm, và vòng lặp quyết định. Bạn chỉ cấp model + tools + system_prompt.
    agent = create_agent(model=llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT)

    print("\n" + "=" * 66)
    print(f"[create_agent] QUESTION: {question}")
    print("=" * 66)

    # 1) invoke — lấy kết quả cuối
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    final = _last_text(result["messages"])
    print(f"[create_agent] FINAL: {final}")

    # 2) stream(stream_mode='updates') — đọc trace từng node:
    #    {node: state} với node 'agent' (AIMessage có thể chứa tool_calls)
    print("\n[create_agent] TRACE (stream_mode='updates'):")
    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]}, stream_mode="updates"
    ):
        for node_name, state in step.items():
            msg = state.get("messages", [])[-1]
            if getattr(msg, "tool_calls", None):
                print(f"  - node '{node_name}' -> gọi tool: {msg.tool_calls}")
            else:
                print(f"  - node '{node_name}' -> text: {_content_to_text(getattr(msg, 'content', None))[:120]}")
    return final


# ============================================================
# PHẦN 2 — StateGraph THỦ CÔNG
# ============================================================
# State: lịch sử chat. Reducer add_messages có nhiệm vụ GỘP tin mới vào list cũ
# (gim cho đúng thứ tự, sửa tin trùng id). Đây là "kít thống" của graph.
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def call_model(state: AgentState):
    """Node 'agent' thủ công: lấy toàn bộ messages trong state, gọi model đã bind_tools.

    Nếu model muốn dùng tool, nó trả AIMessage có tool_calls. Trả về dict dạng
    {"messages":[...]} để reducer add_messages gộp vào state."""
    response = llm.bind_tools(TOOLS).invoke(state["messages"]) # câu lênh này gọi model đã bind_tools, state["messages"] là lịch sử chat. Model có thể trả AIMessage có tool_calls.
    return {"messages": [response]} # Trả về dict {"messages":[...]} để reducer add_messages gộp vào state.


def build_manual_graph() -> StateGraph:
    """Cách 2: tự lắp graph bằng StateGraph.

    Sơ đồ:
        START -> [node 'agent'] --tools_condition-->  'tools' (nếu có tool_calls)
             ^  └──────────────────────────────────────┘
                 'tools' -> quay lại 'agent'
    """
    g = StateGraph(AgentState)
    g.add_node("agent", call_model)                    # bạn tự viết node này
    g.add_node("tools", ToolNode(TOOLS))               # node này từ langgraph.prebuilt
    g.add_edge(START, "agent")
    # tools_condition: trả "tools" nếu AIMessage cuối có tool_calls, ngược lại trả END
    # vì sao biết trả END? Vì nếu không có tool_calls, agent đã trả Final Answer, nên graph kết thúc. 
    # có ví dụ nào khác mà Agent ko trả tool_calls  nhưng vẫn tiếp tục mà ko END ko? 
    # trả lời: không, vì nếu agent ko trả tool_calls thì nó đã trả Final Answer, nên graph kết thúc.
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile()


def run_with_manual_graph(question: str) -> str:
    """Cách 2: chạy graph thủ công (chưa memory)."""
    graph = build_manual_graph()
    print("\n" + "=" * 66)
    print(f"[StateGraph] QUESTION: {question}")
    print("=" * 66)
    full = graph.invoke({"messages": [{"role": "user", "content": question}]})
    final = _last_text(full["messages"])
    print(f"[StateGraph] FINAL: {final}")

    print("\n[StateGraph] trace (stream_mode='updates'):")
    for step in graph.stream(
        {"messages": [{"role": "user", "content": question}]}, stream_mode="updates"
    ):
        for node_name, state in step.items():
            msg = state["messages"][-1]
            print(f"  - node '{node_name}': {_short(msg)}")
    return final


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


def _short(msg) -> str:
    """Tóm tắt một message để in trace gọn."""
    if getattr(msg, "tool_calls", None):
        return f"gọi tool: {[tc['name'] for tc in msg.tool_calls]}"
    return repr(_content_to_text(getattr(msg, "content", msg)))


# ============================================================
# PHẦN 3 — MEMORY ĐA LƯỢT (checkpointer + thread_id)
# ============================================================
def demo_memory_cases():
    """Chứng minh tính nhớ của checkpointer + thread_id.

    Cơ chế: 
      - InMemorySaver lưu toàn bộ State (chứa 'messages') sau mỗi node.
      - Khi invoke kèm configurable.thread_id, LangGraph nạp lại state của THREAD đó
        rồi thêm tin mới vào; nhờ add_messages nên lịch sử được gộp liên tục.
      - Khác thread_id = cuộc chat mới, không thấy lượt trước.
    """
    memory = InMemorySaver()

    # (A) Tạo 2 agent, 1 không memory & 1 có memory (cùng thread_id) để so sánh.
    no_mem_agent = create_agent(model=llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT)
    mem_agent = create_agent(
        model=llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT, checkpointer=memory
    )

    thread = {"configurable": {"thread_id": "cuoc-chat-1"}}

    turn1_q = "Tên tôi là An, 3x4 bằng mấy?"
    turn2_q = "Tên tôi là gì?"

    print("\n" + "=" * 66)
    print("[Memory] KHÔNG memory:")
    r1 = no_mem_agent.invoke({"messages": [{"role": "user", "content": turn1_q}]})
    r2 = no_mem_agent.invoke({"messages": [{"role": "user", "content": turn2_q}]})
    print(f"  lượt 1: {_last_text(r1['messages'])}")
    print(f"  lượt 2: {_last_text(r2['messages'])}   <- (không nhớ 'An')")

    print("-" * 66)
    print("[Memory] CÓ checkpointer + thread_id:")
    ra = mem_agent.invoke({"messages": [{"role": "user", "content": turn1_q}]}, config=thread)
    rb = mem_agent.invoke({"messages": [{"role": "user", "content": turn2_q}]}, config=thread)
    print(f"  lượt 1: {_last_text(ra['messages'])}")
    print(f"  lượt 2: {_last_text(rb['messages'])}   <- nhớ tên An")

    # (B) Cùng memory nhưng thread_id KHÁC → coi như cuộc chat mới.
    other_thread = {"configurable": {"thread_id": "cuoc-chat-2"}}
    rc = mem_agent.invoke({"messages": [{"role": "user", "content": turn2_q}]}, config=other_thread)
    print("-" * 66)
    print(f"[Memory] thread_id KHÁC -> lượt 2: {_last_text(rc['messages'])}   <- không nhớ An")

    # (C) Graph thủ công cũng gắn memory được bằng cách compile(checkpointer=memory).
    graph_mem = build_manual_graph_with_checkpointer()
    rg = graph_mem.invoke({"messages": [{"role": "user", "content": turn1_q}]}, config=thread)
    rg2 = graph_mem.invoke({"messages": [{"role": "user", "content": turn2_q}]}, config=thread)
    print("-" * 66)
    print("[Memory] StateGraph thủ công + InMemorySaver (cùng thread):")
    print(f"  lượt 1: {_last_text(rg['messages'])}")
    print(f"  lượt 2: {_last_text(rg2['messages'])}")


def build_manual_graph_with_checkpointer():
    """Chỉ khác build_manual_graph ở một dòng: compile(checkpointer=memory)."""
    g = StateGraph(AgentState)
    g.add_node("agent", call_model)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=InMemorySaver())


# ============================================================
# CHẠY
# ============================================================
if __name__ == "__main__":
    question = (
        "Dân số Việt Nam nhân đôi rồi cộng dân số Nhật Bản, rồi trừ đi chiều cao đỉnh Everest, kết quả bằng bao nhiêu?"
    )
    # Phần 1 + Phần 2
    # run_with_create_agent(question)
    run_with_manual_graph(question)

    # # Phần 3: memory
    # print("\n" + "=" * 66)
    # print("PHẦN 3 — DEMO MEMORY 2 LƯỢT")
    # print("=" * 66)
    # demo_memory_cases()
    # print("\n[OK] Done.")