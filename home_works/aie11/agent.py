import ast
import json
import operator
import re
import unicodedata

from openai import Client
from pydantic_settings import BaseSettings
from google import genai



class Settings(BaseSettings):
    OPENROUTER_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()

if not settings.OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.")

if not settings.GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
MAX_STEPS = 6

client = Client(
    api_key=settings.OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1"
)

gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)

KNOWLEDGE = {
    "dân số việt nam 2026": "102.3 triệu người (2026)",
    "số thiết bị java 2026": "3 tỷ thiết bị (2026)",
    "gdp việt nam 2024": "430 tỷ USD (2024)",
    "chiều cao đỉnh everest": "8848.86 mét",
    "dân số nhật bản 2025": "123.9 triệu người (2025)",
    "tốc độ ánh sáng": "299792458 mét/giây",
}


def _normalize(text: str) -> str:
    # chuẩn hóa văn bản để so sánh
    # ví dụ: dân số Việt Nam 2026 -> dan so viet nam 2026
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower().strip()


def search(query: str) -> str:
    print(f"tools đang tra cứu dữ kiện cho: {query}")
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
    # kiểm tra 
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
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.10g}"
    return str(value)


def calculator(expr: str) -> str:
    print(f"tools đang thực hiện phép tính cho: {expr}. Ví dụ: 102.3 * 2 ; 100 - 3 * 2 ; etc.")
    try:
        tree = ast.parse(expr, mode="eval")
        return _format_number(_eval_node(tree.body))
    except ZeroDivisionError as e:
        return f"Error: {e}"
    except (SyntaxError, ValueError) as e:
        return f"Error: invalid expression '{expr}' ({e})"


TOOLS = {
    "search": search,
    "calculator": calculator,
}

TOOL_NAMES = ", ".join(TOOLS.keys())


def run_tool(name: str, arg: str) -> str:
    if name not in TOOLS:
        return f"Error: tool '{name}' không tồn tại. Chỉ có: {TOOL_NAMES}."
    try:
        return str(TOOLS[name](arg))
    except Exception as e:
        return f"Error: {e}"


SYSTEM_PROMPT = """You are an AI agent that can use tools to answer questions. You have access to the following tools:
- search(query): tra cứu kho dữ kiện, trả về kết quả hoặc "Không tìm thấy dữ kiện".
- calculator(expr): tính biểu thức số học, ví dụ "102.3 * 2".

QUY TẮC:
1. Mỗi lượt CHỈ được viết ĐÚNG 3 dòng: Thought, Action, Action Input, rồi DỪNG NGAY. Không viết gì thêm.
2. Observation do hệ thống chạy tool rồi cung cấp. KHÔNG bao giờ tự viết "Observation:", không tự kết luận khi chưa có đủ dữ kiện.
3. Chỉ trả "Final Answer" ở lượt mà bạn đã thấy đủ dữ kiện trong Observation.
4. Không dùng JSON, không markdown, không lời dẫn đầu/cuối.

Ví dụ lượt 1:
Question: Dân số Việt Nam nhân đôi bằng bao nhiêu?
Thought: Cần tìm dân số Việt Nam trước.
Action: search
Action Input: dân số Việt Nam

(Chờ hệ thống trả Observation. Lượt tiếp theo lại viết tiếp Thought/Action/Action Input. Khi đủ dữ kiện thì viết "Thought: ...\nFinal Answer: ...")

Question: {{question}}"""


FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)
ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\n?\s*Action Input:\s*(.+)", re.DOTALL)


def _extract_json_field(text: str, key: str) -> str | None:
    m = re.search(rf'"{re.escape(key)}"\s*:\s*["\']([^"\']*)["\']', text)
    return m.group(1).strip() if m else None


def parse_response(text: str) -> tuple[str, ...]:
    """ llm response có thể là 1 trong 3 loại:
    # 1. final answer: ("final", final_answer)
    # 2. action: ("action", tool_name, tool_arg)
    # 3. error: ("error", text)
    # """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text).rstrip("`").strip()
    brace = text.find("{")
    if brace > 0:
        text = text[brace:]
        end = text.rfind("}")
        if end != -1:
            text = text[: end + 1]

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            norm = {
                str(k).strip().lower().replace(" ", "_"): v
                for k, v in data.items()
            }
            if "final_answer" in norm:
                return ("final", str(norm["final_answer"]).strip())
            if "action" in norm and "action_input" in norm:
                return (
                    "action",
                    str(norm["action"]).strip(),
                    str(norm["action_input"]).strip(),
                )
    except (json.JSONDecodeError, ValueError):
        pass

    for key in ("Action Input", "action_input", "actionInput"):
        arg = _extract_json_field(text, key)
        if arg:
            name = _extract_json_field(text, "Action") or _extract_json_field(
                text, "action"
            )
            if name:
                return ("action", name, arg)
    final = _extract_json_field(text, "Final Answer") or _extract_json_field(
        text, "final_answer"
    )
    if final:
        return ("final", final)

    m = FINAL_RE.search(text)
    if m:
        return ("final", m.group(1).strip())
    m = ACTION_RE.search(text)
    if m:
        return ("action", m.group(1).strip(), m.group(2).strip())
    return ("error", text)


def truncate_response(text: str) -> str:
    print("***** [DEBUG] truncate_response *****")
    print(f"Original text: {text}")
    print("***** *****")
    # hàm này thực hiện các bước sau:
    # 1. loại bỏ khoảng trắng ở đầu và cuối chuỗi
    # 2. tìm kiếm chuỗi "Observation:" trong văn bản, nếu tìm thấy, cắt bỏ phần văn bản từ vị trí đó trở đi
    # 3. tìm kiếm chuỗi "Action Input:" trong văn bản, nếu tìm thấy, cắt bỏ phần văn bản từ vị trí đó trở đi
    # 4. trả về văn bản đã được cắt
    # 5. loại bỏ khoảng trắng ở đầu và cuối chuỗi trước khi trả về

    text = text.strip()
    idx = text.find("Observation:")
    # thực hiện cắt bỏ phần văn bản từ vị trí "Observation:" trở đi
    if idx != -1:
        text = text[:idx]
    m = re.search(
        r"Action Input:\s*(.*?)(?=\s*(?:Thought:|Action:|Final Answer:|$))",
        text,
        re.DOTALL,
    )
    if m:
        text = text[: m.end()]
    return text.strip()


def call_llm(messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
        stop=["Observation:"],
    )
    
    print("***** [DEBUG] call_llm *****")
    print(f"LLM response: {response.choices[0].message.content}")
    print("***** *****")
    return response.choices[0].message.content or ""


def build_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(question=question)},
        {"role": "user", "content": question},
    ]


def print_separator(char: str = "=", length: int = 60) -> None:
    print(char * length)


def run_agent(question: str) -> str:
    messages = build_messages(question)
    print(f"\n>>> QUESTION: {question}")
    print_separator()

    for step in range(MAX_STEPS):
        print(f"\n----- STEP {step + 1}/{MAX_STEPS} -----")
        text = truncate_response(call_llm(messages))
        
        # vì sao lại append thêm {"role": "assistant", "content": text} vào messages?
        # role assistant chính là llm, lưu trữ các phản hồi của llm, để llm có thể dựa vào các phản hồi trước đó để đưa ra quyết định tiếp theo
        messages.append({"role": "assistant", "content": text})
        print(text)

        kind = parse_response(text)
        if kind[0] == "final":
            print(f"\n[!] Model output final answer: {kind[1]}")
            return kind[1]
        if kind[0] == "error":
            # print thử text trong tuple kind, để debug xem text có đúng định dạng ReAct không
            print(f"\n[!] Model output error Output: {kind[1]}")
            print(f"\n[!] Model output không theo đúng định dạng ReAct.")
            messages.append({"role": "user", "content": f"Error: {kind[1]}"})

        if kind[0] == "action":
            print(f"\n[!] Model output action: {kind[1]}")
            # _ là gì? tool_name là gì? tool_arg là gì?
            # _ (gạch dưới) đại diện cho phần đầu tiên của tuple, nhưng không được sử dụng, nên ta bỏ qua, phần đầu tiên của tuple là gì mà bỏ qua? l
            _, tool_name, tool_arg = kind
            observation = run_tool(tool_name, tool_arg)
            print(f"Observation: {observation}")
            messages.append(
                {"role": "user", "content": f"Observation: {observation}"}
            )

    return "chưa xong"


def main() -> None:
    questions = [
        # "Số thiết bị Java vào năm 2026 là bao nhiêu?",
        "Dân số Việt Nam 2026 nhân đôi rồi cộng dân số Việt Nam 2026 bằng bao nhiêu?",
    ]
    for question in questions:
        answer = run_agent(question)
        print_separator()
        print(f"FINAL ANSWER: {answer}")
        print_separator()


if __name__ == "__main__":
    main()
