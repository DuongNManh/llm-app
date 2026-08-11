"""Langgraph dựng lại bài tập 11, convert từ thủ công sang LangGraph (agent + ToolNode + tools_condition) + memory checkpointer."""

## dùng create_agent() để tạo agent, bind các tool + llm, + prompt
import ast
import operator
import unicodedata

from pydantic_settings import BaseSettings
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


class Settings(BaseSettings):
    GEMINI_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_API_KEY_2: str | None = None
    TAVILY_API_KEY: str | None = None
    model_config = {"env_file": ".env", "extra": "ignore"}
    
settings = Settings()

MODEL = "gemini-3.1-flash-lite"

llm = ChatGoogleGenerativeAI(
    model = MODEL,
    api_key=settings.GEMINI_API_KEY,
    temperature=0.0
)

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


def search(query: str) -> str:
    """Tra cứu dữ liệu đã biết (dân số, GDP,...). Trả về dữ liệu khớp
    với query, hoặc thông báo không tìm thấy"""
    norm_q = _normalize(query)
    for key, value in KNOWLEDGE.items():
        norm_key = _normalize(key)
        if norm_q in norm_key or norm_key in norm_q:
            return f"{key}: {value}"
    return f"Không tìm thấy dữ kiện cho: {query}"


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

# setup prompt - vì create_agent phiên bản mới đã tự nạp Thought - Action - Observation rồi
# nên ta cần 1 text prompt thuần là đủ

SYSTEM_PROMPT = (
    "You are an agent answer the question best as you can, you can using following tools"
    "search(query) tool when you need a REAL DATA (population, GDP, number,...) "
    "calculator(expression) tool for performing calculate operation in the expression "
    "Answer in Vietnamese and consitence"
)

## ======= Phần 1: create_agent =========
from langchain.agents import create_agent

def run_with_create_agent(question: str) -> str:
    """framework dựng toàn bộ graph cho ta rồi"""
    # từ ReAct template (Thought -> Action -> Observation -> if -> for loop)
    # create_agent setup sẵn agent_node (llm) + (bind_tools), 
    # tools_node (ta cấp tool và create_agent tạo Tools Node)
    # và vòng for THOUGHT -> ACTION -> OBSERVATION đã được tự động
    
    # trong create_agent, framework có lo cho ta AgentState ko? 
    # trả lời: có, create_agent sẽ tự động quản lý trạng thái của agent, bao gồm lịch sử các message (Thought, Action, Observation) và các thông tin liên quan đến trạng thái của các tool. AgentState được tạo ra và duy trì bởi framework, giúp agent có thể tiếp tục cuộc hội thoại một cách mạch lạc và nhất quán.
    
    # nhưng mình muốn hỏi là state phức tạp hơn như có thêm context, memory, kết quả tool call,... thì create_agent có lo cho ta ko?
    # trả lời: create_agent chủ yếu tập trung vào quản lý lịch sử message và trạng thái cơ bản của agent (như là các tool đã được bind). Nếu bạn muốn lưu trữ thêm context, memory, hoặc kết quả từ các tool call, bạn có thể cần triển khai các cơ chế bổ sung hoặc mở rộng AgentState để lưu trữ những thông tin này.
    agent = create_agent(
        tools=TOOLS,
        model=llm,
        system_prompt=SYSTEM_PROMPT,
    )

    print("\n" + "=" * 66)
    print(f"[create_agent] QUESTION: {question}")
    print("=" * 66)

    # 1) invoke lấy kết quả cuối cùng (Final Answer)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    final = _last_text(result["messages"])
    print(f"[create_agent] FINAL: {final}")

    # 2) stream (stream_mode="update") để in trace Thought -> Action -> Observation
    print("\n" + "=" * 66)
    print("\n[create_agent] STREAM (Thought -> Action -> Observation):")
    for step in agent.stream({
        "messages": [{"role": "user", "content": question}]}, stream_mode="updates"):
        for node_name, state in step.items():
            msg = state.get("messages", [])[-1]
            if getattr(msg, "tool_calls", None):
                print(f"  - node '{node_name}' -> gọi tool: {msg.tool_calls}")
            else:
                print(f"  - node '{node_name}' -> text: {_short(_content_to_text(getattr(msg, 'content', None)))}")
    return final



## ======= phần 2: tạo thủ công graph (agent_node + tools_node + tools_condition) =========
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages   # reducer: gộp lịch sử chat
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict


# AgentState thực chất là gì? vì sao dùng TypedDict cho AgentState?
    # AgentState như kho lưu trữ trạng thái của agent trong quá trình chat
    # không chỉ lưu trữ các message (lịch sử chat) mà còn có thể lưu trữ các thông tin khác như trạng thái của các tool, kết quả trung gian, hoặc bất kỳ dữ liệu nào mà agent cần để đưa ra quyết định.
    # TypedDict là dictionary có kiểu dữ liệu cụ thể cho các key, giúp kiểm tra kiểu dữ liệu khi lập trình, đảm bảo rằng các giá trị lưu trữ trong AgentState có đúng kiểu dữ liệu mong muốn.
    # việc sử dụng TypedDict cho AgentState giúp tăng tính rõ ràng và an toàn khi làm việc với trạng thái của agent, đặc biệt là khi có nhiều thông tin cần lưu trữ và quản lý.
    # ví dụ cho Agent + mini RAG: AgentState cần gì?
        # AgentState có thể lưu trữ các thông tin sau:
        # - messages: danh sách các message đã trao đổi giữa user và agent
        # - tool_results: kết quả từ các tool đã được gọi
        # - context: thông tin ngữ cảnh hoặc dữ liệu bổ sung mà agent cần để đưa ra quyết định
        # - memory: thông tin lưu trữ dài hạn mà agent có
class AgentState(TypedDict):
    messages: Annotated[list, add_messages] # reducer add_messages của LangGraph gộp các message vào list messages

def call_model(state: AgentState):
    """node agent thủ công: lấy messages từ AgentState, gọi llm đã bind tools, trả về message mới (Thought)
    Nếu model muốn dùng tool, nó trả AIMessage có tool_calls:
    {
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id":"call_1",
                    "name":"weather",
                    "args":{
                        "city":"Da Nang"
                    }
                }
            ]
    }
    message trả về dù có content hay ko có được gộp vào AgentState hay ko? => gộp vào AgentState.messages, vì sao? vì Thought là tư duy của agent, nó có thể có content (tư duy) hoặc không có content (chỉ gọi tool), nhưng dù sao cũng là AIMessage nên gộp vào AgentState.messages để lưu lịch sử chat.
    """
    # sau Thought này, Action là do tool node chạy? đúng ko? 
    # trả lời: đúng, sau khi node agent tạo ra Thought (tư duy), Action sẽ được thực hiện bởi các tool node dựa trên Thought đó. Thought sẽ xác định xem cần gọi tool nào và với tham số gì, sau đó tool node sẽ thực hiện Action và trả về kết quả (Observation) cho agent.
    response = llm.bind_tools(TOOLS).invoke(input=state["messages"])
    # vì sao là invoke(input=state["messages"]) mà không phải invoke(input=state)? vì state là AgentState, nó có nhiều key, nhưng llm chỉ cần messages để tạo Thought, nên chỉ truyền messages vào invoke.
    # khi invoke lần đầu tiên, state["messages"] chỉ có 1 message là HumanMessage (user hỏi)
    # khi invoke lần thứ 2, state["messages"] có các message là HumanMessage (user hỏi) + AIMessage (Thought của agent), llm sẽ dựa vào 2 message này để tạo Thought tiếp theo.

    # response llm trả về là AIMessage, có dạng sau:
    # response = AIMessage(
    #content="",
    #tool_calls=[
        #{
            #"id":"call_1",
            #"name":"weather",
            #"args":{
                #"city":"Da Nang"
            #}
        #}
    #]
    #)
    # có thể thấy theo ví dụ, tool_calls đang cần thực hiện gọi tool "weather" arg = {"city":"Da Nang"} thì làm thế nào các node tool biết mà sử dụng? => tools_condition sẽ check tool_calls của AIMessage, nếu có tool_calls thì gọi tool node tương ứng, nếu không có tool_calls thì không gọi tool node.
    # trả về dict {"messages": [...]} để gộp vào AgentState.messages
    return {"messages": [response]} # AIMEssage là response do llm.bind_tools(TOOLS).invoke() trả về, có thể có tool_calls hoặc content, dù là tool_calls hay là content thì đây đều là AIMessage (THOUGHT) của agent, nên luôn gộp vào AgentState.messages để lưu lịch sử chat.

def build_manual_graph() -> StateGraph[AgentState]:
    """luồng đi của demo
                Agent
                    │
                    ▼
            Có tool_calls?
                /           \
            YES            NO
            │              │
            ▼              ▼
            ToolNode         END
            │
            │ Observation
            ▼
            Agent
            │
            ▼
        Có tool_calls?
            ...
    """
    g = StateGraph(AgentState)
    g.add_node("agent", call_model) # node agent, llm mình đã bind trước 2 tool search + calculator, khi agent node chạy, nó tạo ra THOUGHT (AIMessage) có thể có tool_calls hoặc không, nếu có tool_calls thì đi qua tools_condition edge để gọi tool node, nếu không có tool_calls thì đi thẳng về agent node.
    g.add_node("tools", ToolNode(TOOLS)) # node tools, khi chạy, nó sẽ thực hiện tool_calls của Thought (AIMessage) mà agent node tạo ra, và trả về kết quả Observation (AIMessage) cho agent node.
    g.add_edge(START, "agent") # node START là node khởi đầu, khi user hỏi (invoke), HumanMessage sẽ được gộp vào AgentState.messages, và đi vào node agent đầu tiên vì ta setup nó là node đầu tiên.
    g.add_conditional_edges("agent", tools_condition)# conditional edge từ agent node, nếu AIMessage (THOUGHT + ACTION) có tool_calls thì đi qua tools_condition edge để gọi tool node, nếu không có tool_calls thì đi về END (kết thúc). 
    # tools_condition là hàm kiểm tra tool_calls của AIMessage, nếu có tool_calls thì trả về True, nếu không có tool_calls thì trả về False.
    # vì sao đi về END khi không có tool_calls? vì khi THOUGHT + ACTION không có tool_calls, nghĩa là agent đã có câu trả lời cuối cùng (Final Answer), nên không cần gọi tool node nữa, và kết thúc luồng xử lý.
    g.add_edge("tools", "agent") # edge từ tools node về agent node, khi tool node thực hiện xong, nó trả về Observation (AIMessage) cho agent node, và agent node sẽ tiếp tục tạo Thought mới dựa trên Observation đó.
    return g.compile()

def run_with_manual_grpah(question: str) -> str:
    """Build LangGraph graph thủ công, chạy graph và in trace Thought -> Action -> Observation"""
    graph = build_manual_graph()
    # state: AgentState = {"messages": [{"role": "user", "content": question}]}
    print("\n" + "=" * 66)
    print(f"[manual_graph] QUESTION: {question}")
    print("=" * 66)
    # 1) invoke lấy kết quả cuối cùng (Final Answer)
    result = graph.invoke({"messages": [{"role": "user", "content": question}]})
    final = _last_text(result["messages"])
    print(f"[manual_graph] FINAL: {final}")
    # 2) stream (stream_mode="update") để in trace Thought -> Action -> Observation
    print("\n" + "=" * 66)
    print("\n[manual_graph] STREAM (Thought -> Action -> Observation):")
    for step in graph.stream(
        {'messages' : [{"role": "user", "content": question}]},
        stream_mode = "updates"
    ):
        for node_name, state in step.items():
            msg = state["messages"][-1]
            print(f"  - node '{node_name}': {msg}")
    return final

## ===== phần 3: MEMORY Đa lượt =========
# giúp ta invoke nhiều lần mà vẫn giữ được context (lịch sử chat) 
from langgraph.checkpoint.memory import InMemorySaver

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

## ===== phần 4: helper functions =========

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
    return repr(_content_to_text(getattr(msg, "content", msg)))[:250]


## =====================
#  Chạy thử các demo
## ======================  
def main():
    question = "Dân số Việt Nam 2026 nhân đôi cộng dân số nhật bản 2025 nhân 3 bằng bao nhiêu?"
    # run_with_create_agent(question)
    # run_with_manual_grpah(question)
    demo_memory_cases()


if __name__ == "__main__":
    main()