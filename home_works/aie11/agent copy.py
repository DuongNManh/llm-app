import ast
import json
import operator
import re
import time
import unicodedata

from openai import Client
from pydantic_settings import BaseSettings
from google import genai



class Settings(BaseSettings):
    OPENROUTER_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()

print(f"OPENROUTER_API_KEY: {settings.OPENROUTER_API_KEY}")
print(f"GEMINI_API_KEY: {settings.GEMINI_API_KEY}")

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
    print(f"tools đang thực hiện phép tính cho: {expr}")
    try:
        tree = ast.parse(expr, mode="eval")
        return _format_number(_eval_node(tree.body))
    except ZeroDivisionError as e:
        return f"Error: {e}"
    except (SyntaxError, ValueError) as e:
        return f"Error: invalid expression '{expr}' ({e})"

# alias tools để gọi các hàm search và calculator
TOOLS = {
    "search": search,
    "calculator": calculator,
}

TOOL_NAMES = ", ".join(TOOLS.keys())


def run_tool(name: str, arg: str) -> str:
    if name not in TOOLS:
        return f"Error: tool '{name}' không tồn tại. Chỉ có: {TOOL_NAMES}."
    try:
        # vì sao python hỗ trợ được kiểu gọi này nhỉ?
        # vì TOOLS[name] là một hàm, và arg là tham số truyền vào hàm đó
        # nghĩa là mình truyền name là search thì đang alias qua hàm search đúng ko?
        # trả lời cho câu hỏi trên: đúng vậy, TOOLS[name] sẽ trả về hàm tương ứng với tên tool, và arg là đối số được truyền vào hàm đó. Khi gọi TOOLS[name](arg), Python sẽ thực hiện gọi hàm tương ứng với tên tool và truyền arg làm đối số.
        return str(TOOLS[name](arg))
    except Exception as e:
        return f"Error: {e}"


SYSTEM_PROMPT = """Bạn là một agent dùng tool để trả lời câu hỏi. Bạn có 2 tool:
- search(query): tra cứu kho dữ kiện, trả về kết quả hoặc "Không tìm thấy dữ kiện".
- calculator(expr): tính biểu thức số học, ví dụ "102.3 * 2".

Bạn CHỈ được xuất 1 trong 2 dạng sau (không JSON, không markdown, không lời giải thích ngoài các dòng này):

DẠNG A — khi cần dữ kiện hoặc phải tính toán. LUÔN đúng 3 dòng:
Thought: <lý do ngắn>
Action: <search | calculator>
Action Input: <chuỗi>

DẠNG B — khi đã có đủ dữ kiện từ Observation. LUÔN đúng 2 dòng:
Thought: <tóm tắt>
Final Answer: <câu trả lời thật>

QUY TẮC:
1. TUYỆT ĐỐI không tự tính toán hay tự suy luận số học trong đầu. Mọi phép tính bắt buộc gọi calculator; mọi dữ kiện bắt buộc lấy qua search.
2. Dạng A có đúng 3 dòng, Dạng B có đúng 2 dòng. Không trộn lẫn, không thêm dòng nào khác.
3. KHÔNG bao giờ tự viết "Observation:". Hệ thống chạy tool rồi cung cấp Observation cho bạn.
4. "Final Answer:" phải là câu trả lời thật, không phải giải thích hay lặp lại quy tắc.

Ví dụ:
Question: Dân số Việt Nam nhân đôi bằng bao nhiêu?
Thought: Cần tìm dân số Việt Nam trước.
Action: search
Action Input: dân số Việt Nam

(lượt mới, sau khi nhận Observation từ hệ thống)
Thought: Dân số 102.3 triệu, cần nhân đôi.
Action: calculator
Action Input: 102.3 * 2

(lượt mới, sau khi nhận Observation từ hệ thống)
Thought: Đã có đủ dữ kiện.
Final Answer: Dân số Việt Nam nhân đôi là 204.6 triệu người.

Bắt đầu!

Question: {{question}}"""


FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)
ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\n?\s*Action Input:\s*(.+)", re.DOTALL)


def _extract_json_field(text: str, key: str) -> str | None:
    m = re.search(rf'"{re.escape(key)}"\s*:\s*["\']([^"\']*)["\']', text)
    return m.group(1).strip() if m else None


def parse_response(text: str) -> tuple[str, ...]:
    """ llm response có thể là 1 trong 3 loại:
        1. final answer: ("final", final_answer)
        2. action: ("action", tool_name, tool_arg)
        3. error: ("error", text)
    """
    text = text.strip()
    # xử lý dấu ``` ở đầu và cuối nếu có
    if text.startswith("```"):
        # thay thế ```[language] ở đầu bằng rỗng, và loại bỏ ``` ở cuối
        # kết quả là chỉ còn lại nội dung bên trong ```   ```
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text).rstrip("`").strip()
    # tìm dấu { đầu tiên để xác định phần JSON, nếu có
    brace = text.find("{")
    if brace > 0:
        # cắt text từ dấu { đầu tiên đến hết, và tìm dấu } cuối cùng để xác định phần JSON
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
    print(f">> Original text: \n{text}")
    print("***** *****")
    # strip text để loại bỏ các khoảng trắng thừa
    text = text.strip()
    # tìm index của chuỗi "Observation:"
    idx = text.find("Observation:")
    # nếu idx = -1 thì không tìm thấy, giữ nguyên text
    if idx != -1:
        # nếu idx != -1 thì cắt text từ đầu đến idx
        # nhằm loại bỏ phần Observation giả từ llm tự bịa
        text = text[:idx]
    # tìm "Action Input:" trong text, lấy phần sau nó cho đến trước "Thought:", "Action:", "Final Answer:" hoặc hết text
    m = re.search(
        r"Action Input:\s*(.*?)(?=\s*(?:Thought:|Action:|Final Answer:|$))",
        text,
        re.DOTALL,
    )
    # nếu tìm thấy thì cắt text từ đầu đến m.end() để loại bỏ phần Action Input giả từ llm tự bịa
    if m:
        text = text[: m.end()]
    return text.strip()


def call_llm(messages: list[dict], retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
                stop=["Observation:"],
            )
            # API có thể trả về object lỗi với choices=None (model hết khả dụng / rate-limit)
            if not response.choices:
                print(
                    f"[!] API phản hồi không có 'choices' "
                    f"(model không khả dụng hoặc bị rate-limit): {response}"
                )
                time.sleep(2)
                continue
            content = response.choices[0].message.content or ""
            print("***** [DEBUG] call_llm *****")
            print(f">> LLM response: \n{content}")
            print("***** *****")
            return content
        except Exception as e:
            print(f"[!] Lỗi gọi LLM (lần {attempt + 1}/{retries}): {type(e).__name__}: {e}")
            time.sleep(2)
    # hết retry: trả chuỗi rỗng để run_agent rơi vào nhánh error -> retry ở mức ReAct
    print("[!] Đã hết số lần retry gọi LLM, trả về chuỗi rỗng.")
    return ""


def build_messages(question: str) -> list[dict]:
    """ xây dựng system prompt và user prompt cho LLM """
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(question=question)},
        {"role": "user", "content": question},
    ]


def print_separator(char: str = "=", length: int = 60) -> None:
    print(char * length)


def run_agent(question: str) -> str:
    # bước 1: hỏi chi thì hỏi, ta phải có message dô llm đúng ko?
    messages = build_messages(question)
    print(f"\n>>>> QUESTION: {question}\n")
    print_separator()

    # bước 2: for loop đến MAX_STEPS
    for step in range(MAX_STEPS):
        print_separator()
        print(f"STEP: {step + 1}")
        print_separator()
        # gọi llm để lấy response
        response = call_llm(messages)
        # cắt response để lấy phần cần thiết, loại bỏ phần Observation giả từ llm tự bịa
        truncated_response = truncate_response(response)
        # làm thế nào để llm nhớ được lịch sử chat? append truncated_response vào messages dưới role assistant
        # cơ chế của chat llm trên mạng có tương tự vậy ko nhỉ?
        messages.append({"role": "assistant", "content": truncated_response})
        
        # ta parse response để lấy action hoặc final answer
        # dùng biến kind hứng để tí đem đi nhận diện loại response
        kind = parse_response(truncated_response)
        
        # trường hợp 1: nếu kind[0] == "final" nghĩa là 
        # llm đã done task và trả về final answer
        if kind[0] == "final":
            print(f"\n[!] Model output final answer: {kind[1]}")
            return kind[1]
        # trường hợp 2: nếu kind[0] == "action" nghĩa là
        # llm muốn gọi tool để lấy dữ kiện hoặc tính toán
        if kind[0] == "action":
            print(f"\n[!] Model output action: {kind[1]}")
            holder, tool_name, tool_arg = kind
            # gọi tool để lấy kết quả
            observation = run_tool(tool_name, tool_arg)
            print(f"Observation: {observation}")
            # thực hiện gắn observation vào messages dưới role user
            messages.append({"role": "user","content": f"Observation: {observation}",})
            print_separator()
        # trường hợp 3: nếu kind[0] == "error" nghĩa là
        # llm trả về response không parse được (không phải action, không phải final answer)
        # thay vì bỏ cuộc, ta gửi phản hồi lỗi vào messages để llm tự sửa và retry lượt mới
        if kind[0] == "error":
            print(f"\n[!] Model output error: {kind[1][:200]}")
            print(f"[!] Retrying step {step + 1}...")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Error: response của bạn không đúng định dạng. "
                        "Hãy trả lời lại, chỉ đúng 1 trong 2 dạng:\n"
                        "- Dùng tool: Thought: ...\nAction: <search|calculator>\nAction Input: <input>\n"
                        "- Trả lời: Thought: ...\nFinal Answer: <answer>\n"
                        "Không dùng JSON, không markdown, không lời dẫn."
                    ),
                }
            )
            continue

    return "chưa xong"


def main() -> None:
    questions = [
        # "Số thiết bị Java vào năm 2026 là bao nhiêu?",
        "Dân số Việt Nam vào năm 2026 nhân đôi rồi cộng số thiết bị Java vào năm 2026 bằng bao nhiêu?",
    ]
    for question in questions:
        answer = run_agent(question)
        print_separator()
        print(f"FINAL ANSWER: {answer}")
        print_separator()


if __name__ == "__main__":
    main()
