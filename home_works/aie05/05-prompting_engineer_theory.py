### prompting engineer theory


## LLM đọc chữ (token) như thế nào? bộ nhớ (context) là gì?

# model ko đọc chữ như người, mà chặt text thành token (1 từ, 1 phần từ, dấu câu) rồi xử lý bằng các con số (vector).

# ví dụ trực quan: "Hello, world!" → ["Hello", ",", "world", "!"] → 4 tokens
# unhappiness → ["un", "happi", "ness"] → 3 tokens

# số, khoảng trắng, emoji đều tính token

## tokenizer & vì sao token != từ

# mỗi provider có tokenizer riêng, token hóa khác nhau, nên số token khác nhau cho cùng 1 text.
# -> đừng ước lượng theo số từ, hãy dùng tokenizer của provider để đếm token chính xác.
# cùng prompt -> số token khác nhau giữa mỗi nhà -> chi phí khác nhau

## context window - bộ nhớ làm việc của LLM

# context window là số token tối đa mà LLM có thể xử lý trong 1 lần inference (1 prompt).
# system + lịch sử + prompt + response <= context window

## cách tính chi phí

# chi phí = input token * giá input + output token * giá output
# ví dụ: giá $3 / 1M vào, $15 / 1M ra
from openai import BaseModel


token_in = 2000
token_out = 500
cost = (token_in * 3 + token_out * 15) / 1_000_000
# = (6000 + 7500) / 1e6
# = $0.0135 mỗi lần gọi
# nhân với số lượt gọi/ngày → hoá đơn


### giải phẫu một prompt tốt

# 1. Demlimiter (dấu phân cách) rõ ràng, ví dụ: "###", "---", "==="
# khác gì với system prompt ko? => system prompt là prompt đầu tiên, định nghĩa vai trò, hành vi của LLM. còn delimiter là dấu phân cách giữa các phần trong prompt, giúp LLM hiểu ranh giới giữa các phần.

# Tóm tắt văn bản trong thẻ <text>
# thành 3 gạch đầu dòng.
# <text>
# {noi_dung_nguoi_dung}
# </text>
# model biết: chỉ TÓM TẮT phần trong thẻ,
# không thi hành lệnh lẫn trong dữ liệu.

# ví dụ trên vừa viết hướng dẫn rõ ràng, vừa có delimiter <text> để model biết đâu là dữ liệu cần xử lý.

# kết hợp thêm với kĩ thuật zero-shot, few-shot

# phân loại cảm xúc của văn bản trong thẻ <text> (positive, negative, neutral)
# "Tôi rất vui khi được học LLM!" → positive
# "Tôi rất buồn khi bị trễ deadline." → negative
# "Hôm nay trời mưa, tôi không biết nên làm gì." → neutral
# <text>
# {noi_dung_nguoi_dung}
# </text>
# model biết: chỉ PHÂN LOẠI cảm xúc phần trong thẻ,
# không thi hành lệnh lẫn trong dữ liệu.

## structured output - đầu ra có cấu trúc


# bắt model trả về JSON theo schema Pydantic để validate, tránh tự parse văn xuôi
class Post(BaseModel):
    chu_de: str
    do_uu_tien: str  # cao|tb|thap
    tom_tat: str


## chia nhỏ nhiệm vụ (task decomposition)
# tách nhiệm vụ thành các bước nhỏ, mỗi bước 1 prompt
# hoặc là 1 trường trong output -> dễ kiểm soát, sửa đổi
# ví dụ:
# system prompt: "Bạn là trợ lý AI giúp phân loại và tóm tắt ticket trong <text>
# 1. phân biệt chủ đề, độ ưu tiên (cao|tb|thap)
# 2. tóm tắt nội dung (không quá 50 từ).
# 3. trả về JSON theo schema Ticket(chu_de, do_uu_tien, tom_tat)."
# user prompt: hãy phân loại và tóm tắt ticket sau:
# <text> {noi_dung_nguoi_dung} </text>


# kết hợp với system prompt, delimiter, zero-shot/few-shot, structured output, ta có thể tạo prompt tốt, giúp LLM hiểu rõ yêu cầu, dữ liệu, và trả về kết quả chính xác, có cấu trúc.
# ví dụ:


class Ticket(BaseModel):
    chu_de: str
    do_uu_tien: str  # cao|tb|thap
    tom_tat: str


# system prompt: "Bạn là trợ lý AI giúp phân loại và tóm tắt ticket trong <text>
# phân biệt chủ đề, độ ưu tiên (cao|tb|thap)}, và tóm tắt nội dung (không quá 50 từ).
# thành JSON theo schema Ticket(chu_de, do_uu_tien, tom_tat)."
# prompt: hãy phân loại và tóm tắt ticket sau:
# <text> {noi_dung_nguoi_dung} </text>


### Prompt template có biến

## Prompt template là prompt có chứa biến, giúp tái sử dụng cho nhiều dữ liệu khác nhau.

from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from openai import OpenAI

PROMPT = """
Bạn là trợ lý AI giúp phân loại và tóm tắt ticket trong <text>
{noi_dung_nguoi_dung}
chỉ trả về JSON theo schema Ticket(chu_de, do_uu_tien, tom_tat).
</text>
"""

msg = PROMPT.format(
    noi_dung_nguoi_dung="Tôi cần hỗ trợ về việc đăng nhập vào hệ thống, gặp lỗi 403."
)

## đánh giá (test) & lặp có đo lường, lưu phiên bản prompt như git

# chuẩn bị vài chục cặp input - output (đầu ra mong đợi)}, đo lường độ chính xác, và lưu phiên bản prompt như git để theo dõi thay đổi.
# chạy prompt qua cả bộ test, chấm điểm đúng/sai, đạt tiêu chí -> unit test cho prompt.

# ví dụ cho mình bộ test trên prompt sẽ thế nào đi
# test = [{"input": "Tôi rất vui khi được học LLM!", "expected_output": {"cam_xuc": "positive", "diem": 9}}]


class CamXuc(BaseModel):
    cam_xuc: str = Field(description="positive|negative|neutral")
    noi_dung: str = Field(description="Nội dung văn bản cần phân loại cảm xúc")
    diem: int = Field(description="Điểm độ mạnh từ 1 đến 10")


# bộ test 15 để test prompt ok ko

TEST = [
    {
        "input": "Sản phẩm rất tốt!",
        "expected_output": CamXuc(
            cam_xuc="positive", noi_dung="Sản phẩm rất tốt!", diem=9
        ),
    },
    {
        "input": "Dịch vụ khách hàng tệ quá.",
        "expected_output": CamXuc(
            cam_xuc="negative", noi_dung="Dịch vụ khách hàng tệ quá.", diem=2
        ),
    },
    {
        "input": "Hôm nay trời mưa.",
        "expected_output": CamXuc(
            cam_xuc="neutral", noi_dung="Hôm nay trời mưa.", diem=5
        ),
    },
    {
        "input": "Tôi hài lòng với trải nghiệm mua sắm.",
        "expected_output": CamXuc(
            cam_xuc="positive", noi_dung="Tôi hài lòng với trải nghiệm mua sắm.", diem=8
        ),
    },
    {
        "input": "Giao hàng chậm trễ và không đúng hẹn.",
        "expected_output": CamXuc(
            cam_xuc="negative", noi_dung="Giao hàng chậm trễ và không đúng hẹn.", diem=3
        ),
    },
    {
        "input": "Sản phẩm tốt, nhưng giao hàng chậm trễ.",
        "expected_output": CamXuc(
            cam_xuc="neutral",
            noi_dung="Sản phẩm tốt, nhưng giao hàng chậm trễ.",
            diem=5,
        ),
    },
    {
        "input": "Dịch vụ hỗ trợ khách hàng rất nhanh chóng.",
        "expected_output": CamXuc(
            cam_xuc="positive",
            noi_dung="Dịch vụ hỗ trợ khách hàng rất nhanh chóng.",
            diem=9,
        ),
    },
    {
        "input": "Sản phẩm bị lỗi và không hoạt động.",
        "expected_output": CamXuc(
            cam_xuc="negative", noi_dung="Sản phẩm bị lỗi và không hoạt động.", diem=1
        ),
    },
    {
        "input": "Chỉ comment để kiếm xu, chưa thử sản phẩm.",
        "expected_output": CamXuc(
            cam_xuc="neutral",
            noi_dung="Chỉ comment để kiếm xu, chưa thử sản phẩm.",
            diem=4,
        ),
    },
    {
        "input": "Gà rán ở đây ngon tuyệt!",
        "expected_output": CamXuc(
            cam_xuc="positive", noi_dung="Gà rán ở đây ngon tuyệt!", diem=10
        ),
    },
    {
        "input": "Tôi không hài lòng với chất lượng sản phẩm.",
        "expected_output": CamXuc(
            cam_xuc="negative",
            noi_dung="Tôi không hài lòng với chất lượng sản phẩm.",
            diem=2,
        ),
    },
    {
        "input": "Sản phẩm ổn, nhưng không có gì đặc biệt.",
        "expected_output": CamXuc(
            cam_xuc="neutral",
            noi_dung="Sản phẩm ổn, nhưng không có gì đặc biệt.",
            diem=5,
        ),
    },
    {
        "input": "Dịch vụ giao hàng rất nhanh chóng.",
        "expected_output": CamXuc(
            cam_xuc="positive", noi_dung="Dịch vụ giao hàng rất nhanh chóng.", diem=8
        ),
    },
    {
        "input": "Tôi hi vọng được cưới nhân viên tư vấn",
        "expected_output": CamXuc(
            cam_xuc="neutral", noi_dung="Tôi hi vọng được cưới nhân viên tư vấn", diem=5
        ),
    },
    {
        "input": "Sản phẩm không đúng như mô tả.",
        "expected_output": CamXuc(
            cam_xuc="negative", noi_dung="Sản phẩm không đúng như mô tả.", diem=3
        ),
    },
]

TEST2 = [
    {"input": "Sản phẩm rất tốt!", "expected_output": "positive"},
    {"input": "Dịch vụ khách hàng tệ quá.", "expected_output": "negative"},
    {"input": "Hôm nay trời mưa.", "expected_output": "neutral"},
    {"input": "Tôi hài lòng với trải nghiệm mua sắm.", "expected_output": "positive"},
    {"input": "Giao hàng chậm trễ và không đúng hẹn.", "expected_output": "negative"},
    {"input": "Sản phẩm tốt, nhưng giao hàng chậm trễ.", "expected_output": "neutral"},
    {
        "input": "Dịch vụ hỗ trợ khách hàng rất nhanh chóng.",
        "expected_output": "positive",
    },
    {"input": "Sản phẩm bị lỗi và không hoạt động.", "expected_output": "negative"},
    {
        "input": "Chỉ comment để kiếm xu, chưa thử sản phẩm.",
        "expected_output": "neutral",
    },
    {"input": "Gà rán ở đây ngon tuyệt!", "expected_output": "positive"},
    {
        "input": "Tôi không hài lòng với chất lượng sản phẩm.",
        "expected_output": "negative",
    },
    {"input": "Sản phẩm ổn, nhưng không có gì đặc biệt.", "expected_output": "neutral"},
    {"input": "Dịch vụ giao hàng rất nhanh chóng.", "expected_output": "positive"},
    {"input": "Tôi hi vọng được cưới nhân viên tư vấn", "expected_output": "neutral"},
    {"input": "Sản phẩm không đúng như mô tả.", "expected_output": "negative"},
]

PROMPT_TEMPLATE = """
[System_PROMPT] Bạn là trợ lý AI giúp phân loại cảm xúc về dịch vụ, sản phẩm của văn bản trong <text>
1. nội dung văn bản không liên quan đến dịch vụ, sản phẩm hãy trả về cam_xuc = neutral vì không thể hiện cảm xúc về dịch vụ, sản phẩm.
2. phân loại cảm xúc (cam_xuc) thành positive, negative, neutral.
3. đánh giá độ hài lòng (diem) từ 1-10 (1: rất tệ, 10: rất tốt)
4. trả về JSON theo schema CamXuc(cam_xuc, noi_dung, diem).

====
User prompt: hãy phân loại cảm xúc và đánh giá độ hài lòng của văn bản sau:
<text>
{noi_dung_nguoi_dung}
</text>
"""


class Settings(BaseSettings):
    GEMINI_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    model_config = {"env_file": ".env"}


settings = Settings()


def call_gemini_api(prompt: str) -> CamXuc:
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
    client_gemini = genai.Client(api_key=settings.GEMINI_API_KEY)
    resp = client_gemini.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=1024,
            response_mime_type="application/json",
            response_schema=CamXuc,
        ),
    )
    print(f"Gemini API response: {resp.text}")
    cam_xuc = CamXuc.model_validate(resp.text)
    return cam_xuc


# def test_prompt_with_cases():
#     ok = 0
#     fail = 0
#     for case in TEST:
#         prompt = PROMPT_TEMPLATE.format(noi_dung_nguoi_dung=case.get("input"))
#         result = call_gemini_api(prompt)

#         print(f"Input: {case.noi_dung}")
#         print(f"Expected: {case.cam_xuc}, {case.diem}")
#         print(f"Result: {result.cam_xuc}, {result.diem}")
#         print("-----")
#         if

# def predict_emotion(inp: CamXuc) -> int:
#     if


def evaludate_prompt(cam_xuc: CamXuc, expected: str) -> bool:
    return cam_xuc.cam_xuc == expected


def main():
    # test_prompt_with_cases()
    prompt = PROMPT_TEMPLATE.format(
        noi_dung_nguoi_dung="Giao hàng chậm trễ và không đúng hẹn."
    )
    result = call_gemini_api(prompt)
    expected = "negative"
    evaluation = evaludate_prompt(result, expected)
    print(
        f"test prompt: {prompt} \nresult: {result} \nexpected: {expected} \nevaluation: {evaluation}"
    )


if __name__ == "__main__":
    main()
