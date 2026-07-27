from typing import Callable

from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from openai import OpenAI
import tiktoken
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception,
)


class CamXuc(BaseModel):
    cam_xuc: str = Field(description="positive|negative|neutral")
    noi_dung: str = Field(description="Nội dung văn bản cần phân loại cảm xúc")
    diem: int = Field(description="Điểm hài lòng từ 1 đến 10")


TEST2 = [
    {"input": "Sản phẩm rất tốt!", "expected_output": "positive"},
    {"input": "Dịch vụ khách hàng tệ quá.", "expected_output": "negative"},
    {"input": "Hôm nay trời mưa.", "expected_output": "neutral"},
    {"input": "Tôi hài lòng với trải nghiệm mua sắm.", "expected_output": "positive"},
    {"input": "Giao hàng chậm trễ và không đúng hẹn.", "expected_output": "negative"},
    {"input": "Sản phẩm tốt, nhưng giao hàng chậm trễ.", "expected_output": "negative"},
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
    {"input": "Tôi hi vọng được cưới nhân viên tư vấn.", "expected_output": "neutral"},
    {"input": "Sản phẩm không đúng như mô tả.", "expected_output": "negative"},
]

PROMPT_TEMPLATE_V1 = """
[System_PROMPT] Bạn là trợ lý AI giúp phân loại cảm xúc về dịch vụ, sản phẩm của văn bản trong <text>
1. phân loại cảm xúc (cam_xuc) thành positive, negative, neutral.
2. đánh giá độ hài lòng (diem) từ 1-10 (1: rất tệ, 10: rất tốt)
3. trả về JSON theo schema CamXuc(cam_xuc, noi_dung, diem).
====
User prompt: hãy phân loại cảm xúc và đánh giá độ hài lòng của văn bản sau:
<text>
{noi_dung_nguoi_dung}
</text>
"""

PROMPT_TEMPLATE_V2 = """[System_PROMPT] Bạn là trợ lý AI giúp phân loại cảm xúc về dịch vụ, sản phẩm của văn bản trong <text>
1. nội dung văn bản phải liên quan đến dịch vụ, sản phẩm, nếu không liên quan hãy trả về cam_xuc = neutral vì không thể hiện cảm xúc về dịch vụ, sản phẩm.
2. một số văn bản có thể mỉa mai, châm biếm, hãy phân loại cảm xúc dựa trên ý nghĩa thực sự của văn bản, không dựa trên từ ngữ.
3. phân loại cảm xúc (cam_xuc) thành positive, negative, neutral.
4. đánh giá độ hài lòng (diem) từ 1-10 (1: rất tệ, 10: rất tốt)
5. trả về JSON theo schema CamXuc(cam_xuc, noi_dung, diem).
ví dụ: 
- sản phẩm tốt => CamXuc(cam_xuc="positive", noi_dung="sản phẩm tốt", diem=9)
- dịch vụ khách hàng tệ => CamXuc(cam_xuc="negative", noi_dung="dịch vụ khách hàng tệ", diem=2)
- hôm nay trời mưa => CamXuc(cam_xuc="neutral", noi_dung="hôm nay trời mưa", diem=5)
====
User prompt: hãy phân loại cảm xúc và đánh giá độ hài lòng của văn bản sau:
<text>
{noi_dung_nguoi_dung}
</text>
"""


class EvaluationResult(BaseModel):
    version: str
    correct_count: int
    fail_count: int
    error_count: int
    total_count: int
    accuracy: float
    avg_tokens: float


class Settings(BaseSettings):
    GEMINI_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    model_config = {"env_file": ".env"}


settings = Settings()
encoder = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    """Count the number of tokens in a given text using tiktoken."""
    return len(encoder.encode(text))


def should_retry(exc: BaseException) -> bool:
    import openai

    if isinstance(
        exc, (openai.RateLimitError, openai.APIConnectionError, openai.APIError)
    ):
        return True

    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=2, min=1, max=15),
    retry=retry_if_exception(should_retry),
)
def call_gemini_api(client: genai.Client, prompt: str) -> CamXuc:
    resp = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=150,
            response_mime_type="application/json",
            response_schema=CamXuc,
        ),
    )

    tokens = count_tokens(prompt)
    print(f"token usage for '{prompt[:25]}': {tokens}")
    if not resp.text:
        raise ValueError("No response from Gemini API.")
    print(f"Gemini API response: {resp.text}")
    cam_xuc = CamXuc.model_validate_json(resp.text)
    return cam_xuc


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=2, min=1, max=15),
    retry=retry_if_exception(should_retry),
)
def call_openrouter_api(client: OpenAI, prompt: str) -> CamXuc:
    reponse = client.chat.completions.create(
        model="tencent/hy3:free",
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "CamXuc", "schema": CamXuc.model_json_schema()},
        },
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    if not reponse.choices or not reponse.choices[0].message.content:
        raise ValueError("No response from OpenRouter API.")
    print(f"token usage for '{prompt[:25]}': {count_tokens(prompt)}")
    print(f"OpenRouter API response: {reponse.choices[0].message.content}")
    cam_xuc = CamXuc.model_validate_json(reponse.choices[0].message.content)
    return cam_xuc


def evaluate_prediction(cam_xuc: CamXuc | None, expected: str) -> bool:
    try:
        if cam_xuc is None:
            return False
        print(f"Evaluating: cam_xuc={cam_xuc.cam_xuc}, expected={expected}")
        result = cam_xuc.cam_xuc == expected
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return False
    return result


def evaluate_prompt(
    version: str, prompt_template: str, test_cases: list[dict], call_api_func: Callable
) -> EvaluationResult:
    correct_count = 0
    fail_count = 0
    error_count = 0
    total_count = len(test_cases)
    total_tokens = 0
    with open(f"evaluation_{version}.txt", "w", encoding="utf-8") as f:
        for index, test in enumerate(test_cases, start=1):
            is_correct = False
            input_text = test["input"]
            expected_output = test["expected_output"]
            prompt = prompt_template.format(noi_dung_nguoi_dung=input_text)
            token_count = count_tokens(prompt)
            total_tokens += token_count
            print(f"Prompt: {prompt[:50]}... | Token: {token_count}")
            try:
                cam_xuc = call_api_func(prompt)
                if cam_xuc is None:
                    print(f"Error: No response for input '{input_text}'")
                    error_count += 1
                    continue
                if evaluate_prediction(cam_xuc, expected_output):
                    correct_count += 1
                    is_correct = True
                else:
                    fail_count += 1
            except Exception as e:
                print(f"Error occurred while processing input '{input_text}': {e}")
                error_count += 1
                continue
            f.write("=" * 60 + "\n")
            f.write(f"TEST CASE {index}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Input          : {input_text}\n")
            f.write(f"Expected       : {expected_output}\n")
            f.write(
                f"Actual         : {cam_xuc.cam_xuc if cam_xuc else 'No response'}\n"
            )
            f.write(f"Result         : {'Correct' if is_correct else 'Wrong'}\n")
            f.write(f"Prompt Tokens  : {token_count}\n")
        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 60 + "\n")

        f.write(f"Total Cases    : {total_count}\n")
        f.write(f"Correct        : {correct_count}\n")
        f.write(f"Wrong          : {fail_count}\n")
        f.write(f"Format Error   : {error_count}\n")
        f.write(f"Accuracy       : {correct_count / total_count * 100:.2f}%\n")
        f.write(f"Average Tokens : {total_tokens / total_count:.2f}\n")
    return EvaluationResult(
        version=version,
        correct_count=correct_count,
        fail_count=fail_count,
        error_count=error_count,
        total_count=total_count,
        accuracy=correct_count / total_count * 100 if total_count > 0 else 0,
        avg_tokens=total_tokens / total_count if total_count > 0 else 0,
    )


def print_evaluation_result(result: EvaluationResult):
    print("\n=== KẾT QUẢ ĐÁNH GIÁ ===")
    print(f"Tổng số ca test: {result.total_count}")
    print(f"Số ca đoán đúng: {result.correct_count}")
    print(f"Số ca đoán sai: {result.fail_count}")
    print(f"Số ca hỏng định dạng: {result.error_count}")
    print(f"Độ chính xác (Accuracy): {result.accuracy:.2f} %")
    print(f"Trung bình số token sử dụng: {result.avg_tokens:.2f}")

    with open(f"evaluation_{result.version}.txt", "a", encoding="utf-8") as f:
        f.write(f"Tổng số ca test: {result.total_count}\n")
        f.write(f"Số ca đoán đúng: {result.correct_count}\n")
        f.write(f"Số ca đoán sai: {result.fail_count}\n")
        f.write(f"Số ca hỏng định dạng: {result.error_count}\n")
        f.write(f"Độ chính xác (Accuracy): {result.accuracy:.2f} %\n")
        f.write(f"Trung bình số token sử dụng: {result.avg_tokens:.2f}\n")


def main_v2():
    run_openrouter_evaluation()
    # run_gemini_evaluation()


if __name__ == "__main__":
    main_v2()


def compare_results(result1: EvaluationResult, result2: EvaluationResult):
    print("\n=== SO SÁNH KẾT QUẢ ===")
    print(
        f"Phiên bản 1: {result1.version}, Độ chính xác: {result1.accuracy:.2f} %, Trung bình token: {result1.avg_tokens:.2f}"
    )
    print(
        f"Phiên bản 2: {result2.version}, Độ chính xác: {result2.accuracy:.2f} %, Trung bình token: {result2.avg_tokens:.2f}"
    )


def run_openrouter_evaluation():
    """chạy đánh giá với API OpenRouter"""
    if not settings.OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.")
    openrouter_client = OpenAI(
        api_key=settings.OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1"
    )

    result1 = evaluate_prompt(
        "V1",
        PROMPT_TEMPLATE_V1,
        TEST2,
        lambda prompt: call_openrouter_api(openrouter_client, prompt),
    )

    print_evaluation_result(result1)

    result2 = evaluate_prompt(
        "V2",
        PROMPT_TEMPLATE_V2,
        TEST2,
        lambda prompt: call_openrouter_api(openrouter_client, prompt),
    )

    print_evaluation_result(result2)

    compare_results(result1, result2)


def run_gemini_evaluation():
    """chạy đánh giá với API Gemini"""
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
    gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)

    result1 = evaluate_prompt(
        "V1",
        PROMPT_TEMPLATE_V2,
        TEST2,
        lambda prompt: call_gemini_api(gemini_client, prompt),
    )

    print_evaluation_result(result1)

    result2 = evaluate_prompt(
        "V2",
        PROMPT_TEMPLATE_V2,
        TEST2,
        lambda prompt: call_gemini_api(gemini_client, prompt),
    )

    print_evaluation_result(result2)

    compare_results(result1, result2)
