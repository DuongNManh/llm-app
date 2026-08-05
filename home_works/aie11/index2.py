from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
from google import genai
from openai import OpenAI


class Settings(BaseSettings):
    GEMINI_API_KEY: str | None = None
    OPENROUTER_API_KEY_2: str | None = None
    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()

if not settings.OPENROUTER_API_KEY_2:
    raise ValueError("OPENROUTER_API_KEY_2 is not set in the environment variables.")

MAX_STEPS = 6

gemini_client = genai.Client(api_key=Settings().GEMINI_API_KEY)
openrouter_client = OpenAI(base_url="https://openrouter.ai/v1", api_key=Settings().OPENROUTER_API_KEY_2)



# tool 1

# tool 2
class CalculatorParams(BaseModel):
    a: int = Field(..., description="The first number for the calculation.")
    b: int = Field(..., description="The second number for the calculation.")
    operation: str = Field(..., description="The operation to perform: add, sub, mul, div, pow.")
def calculator(params: CalculatorParams) -> float:
    """"""
    if params.operation == "add":
        return params.a + params.b
    elif params.operation == "sub":
        return params.a - params.b
    elif params.operation == "mul":
        return params.a * params.b
    elif params.operation == "div":
        if params.b != 0:
            return params.a / params.b
        else:
            raise ValueError("Cannot divide by zero.")
    elif params.operation == "pow":
        return params.a ** params.b
    else:
        raise ValueError("Invalid operation. Supported operations: add, sub, mul, div, pow.")


tools = [
    {
        # tool 1: get product info in restaurant's menu
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform basic arithmetic operations.",
            "strict": True,
            "parameters": CalculatorParams.model_json_schema(),
        },
    }
]

TOOLS = {
    "search": search,
    "calculator": calculator,
}

TOOL_NAMES = ", ".join(TOOLS.keys())

SYSTEM_PROMPT = f"""Answer the following questions as best you can. You have access to the following tools:

search(query): Tra cứu một 'kho dữ kiện' nhỏ và trả về chuỗi kết quả, hoặc 'Không tìm thấy dữ kiện cho: ...'.
calculator(expr): Nhận một biểu thức số học dạng chuỗi (ví dụ '102.3 * 2', '100 + 50') và trả về kết quả tính được.

Use the following format exactly:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{TOOL_NAMES}]
Action Input: the input to the action (string)

... (this Thought/Action/Action Input sequence can repeat until you have enough data)

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Important rules:
- Always use tools to get real data. Never make up numbers or facts from general knowledge.
- Use the search tool FIRST when the question needs data, then use the calculator tool to compute.
- NEVER write 'Observation:' yourself. The system will run the tool and provide the Observation.
- Do not repeat a tool call that already succeeded in a previous step.
- If search returns 'Không tìm thấy dữ kiện', report that you cannot answer.

Example session:

Question: Dân số Việt Nam nhân đôi bằng bao nhiêu?
Thought: I need the population of Vietnam first, then I can double it.
Action: search
Action Input: dân số Việt Nam

Begin!

Question: {{question}}"""

