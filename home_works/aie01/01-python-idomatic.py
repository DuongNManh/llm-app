### ── I. Type hint và Pydantic: ──
# làm sao để dữ liệu vào ra LLM luôn đúng hình dạng ta mong đợi?
# cả 2 giúp code Python dễ đọc, dễ maintain, dễ test, dễ debug hơn


## 1. Type hint:
# giúp IDE và linter kiểm tra kiểu dữ liệu,
# giúp người đọc hiểu rõ hơn về input/output của hàm, class, biến
# tuy nhiên, chỉ là gợi ý lỗi trên IDE, Python vẫn chạy được nếu sai kiểu dữ liệu


def add(a: int, b: int) -> int:  # type hint cho input và output
    return a + b


def fuction(texts: list[str]) -> list[list[str]]:  # type hint cho input và output
    # return texts  # lỗi kiểu dữ liệu, ide sẽ cảnh báo ngay, nhưng Python vẫn chạy được vì type hint chỉ là gợi ý, không bắt buộc
    return [texts]  # đúng kiểu dữ liệu, ide sẽ không cảnh báo lỗi


print(
    fuction(["hihihihi"])
)  # vẫn trả về ['hihihihi'] nhưng ide sẽ cảnh báo lỗi kiểu dữ liệu
print(add(1, 2))  # trả về 3


## 2. ── Generic dựng sẵn & 'X | None' (Union) giúp type hint linh hoạt hơn

# 2.1. built-in generic:
list[str]
dict[str, int]
set[int]
tuple[str, int]


# 2.2. Union type hint:
# X | None là cách viết mới của Union[X, None] (Python 3.10+)
# dùng trong mọi codebase AI hiện đại
def get_first_item(items: str | None) -> str | None:
    return items[0] if items else None


# | None nghĩa là giá trị CÓ THỂ vắng mặt
# - buộc bạn xử lý trường hợp None, tránh lỗi runtime
# ví dụ cho hàm trên
get_first_item("hello")  # trả về 'h'
get_first_item(None)  # trả về None, không lỗi runtime


## 3. ── Pydantic BaseModel:
# giúp validate dữ liệu đầu vào, tự động convert kiểu dữ liệu,
# sai kiểu sẽ raise ValidationError ngay, tránh lỗi runtime, giúp code an toàn hơn
# Type hint chỉ là ghi chú, Pydantic mới là validate dữ liệu thật sự

from pydantic import BaseModel


class Article(BaseModel):
    title: str
    tags: list[str]


# Pydantic auto-generates __init__ với type checking + coercion ở runtime
try:
    a = Article(title="Python cơ bản", tags=["code", "learn"])
    print("OK:", a)
except Exception as e:
    print("Lỗi:", e)

# Test runtime: tags sai kiểu sẽ raise ValidationError
try:
    b = Article(title=1, tags="not_a_list")  # title được cast thành str, tags thì fail
    print("OK:", b)
except Exception as e:
    print("Lỗi tags:", e)


##4. ── Pydantic cho AI: bắt LLM trả JSON đúng cấu trúc ──
# đưa Model (có BaseModel) vào LLM SDK, LLM sẽ trả JSON đúng cấu trúc, Pydantic sẽ validate ngay


class Movie(BaseModel):
    title: str
    year: int
    rating: float


# ví dụ SDK
# response = sdk.request("Get movie info", parseModal=Movie)  # LLM trả về JSON đúng cấu trúc Movie

# Giả lập LLM trả về JSON có nhiều field hơn class ta yêu cầu
llm_response = {
    "title": "Inception",
    "year": 2010,
    "rating": 8.8,
    "director": "Christopher Nolan",  # extra field
    "duration": 148,  # extra field
    "cast": ["DiCaprio", "Watanabe"],  # extra field
}

# Pydantic mặc định bỏ qua extra fields → không lỗi
movie = Movie.model_validate(llm_response)
print("Movie:", movie)
# Movie(title='Inception', year=2010, rating=8.8)

# Nếu LLM trả thiếu field → ValidationError
try:
    Movie.model_validate({"title": "Inception"})  # thiếu year, rating
except Exception as e:
    print("Lỗi thiếu field:", e)

# Nếu LLM trả sai kiểu → bắt lỗi ngay
try:
    Movie.model_validate({"title": "Inception", "year": "not_a_year", "rating": "bad"})
except Exception as e:
    print("Lỗi sai kiểu:", e)

# -> type hint (ghi chú) + pydantic (kiểm tra thật) = dữ liệu AI luôn đúng hình dạng


### II. asyncio:
# chạy nhiều task async cùng lúc
# sử dụng async def để define async function, await để chờ kết quả của async function khác
# sử dụng asyncio.gather() để chạy nhiều async function cùng lúc và chờ tất cả hoàn thành

## 1. async def & await:

import asyncio
import datetime


# định nghĩa một corountine function, có thể gọi nó nhưng không chạy ngay,
# trả về một việc chờ thực thi (coroutine object)
async def fetch_data(q: str):
    await asyncio.sleep(1)  # giả lập call API mất 1 giây
    print(f"Data {q} " + str(datetime.datetime.now()))


## 2. asyncio.gather():
# chạy nhiều async function cùng lúc và chờ tất cả hoàn thành
# cách xử lý 1 batch cùng lúc: 100 email, embed 100 câu,... song song


# thử test 3 fetch_data() cùng lúc
async def main():
    qs = {"query 1", "query 2", "query 3"}
    results = await asyncio.gather(*[fetch_data(q) for q in qs])
    print("Results:", results)


# vì event loop đã chạy trong Jupyter Notebook, nên ta không thể dùng asyncio.run() trực tiếp
# ta await main() để chạy async function trong notebook
await main()


## 3.semaphore:
# giới hạn số lượng task async chạy cùng lúc
# tránh quá tải server hoặc API trả 429 Too Many Requests
# hoặc tốn tiền dồn dập khi gọi API trả phí

sem = asyncio.Semaphore(5)  # 5 vé


async def limited_fetch_data(q: str):
    async with sem:  # chỉ cho phép 5 task chạy cùng lúc
        return await fetch_data(q)


# tại hàm main hoặc hàm
print("Start limited fetch at " + str(datetime.datetime.now()))
# chạy 20 task nhưng tối đa 5 vé cùng lúc
# các task còn lại sẽ chờ đến khi có vé (xếp hàng đợi vé)
await asyncio.gather(*[limited_fetch_data(f"query {i}") for i in range(20)])
print("End limited fetch at " + str(datetime.datetime.now()))


### III. Decorator:
# thêm retry, cache, log,... cho một hàm mà ko đụng code gốc

# là một hàm nhận vào một hàm và trả về hàm mới có thêm hành vi mới
# có thể thêm logic trước hoặc sau khi gọi hàm gốc
# có thể dùng cho các việc phụ như (đo giờ, log, retry, cache,...)
# khỏi code chính, decorator dùng lại cho nhiều hàm khác nhau

import functools
from tenacity import retry, stop_after_attempt, wait_fixed


def measure_time(func):
    @functools.wraps(func)  # giữ nguyên tên và docstring của func
    async def inner(*a, **kw):
        start = datetime.datetime.now()
        result = await func(*a, **kw)  # gọi hàm gốc
        end = datetime.datetime.now()
        print(f"Time taken by {func.__name__}: {end - start}")
        return result  # return kết quả của hàm gốc

    return inner


def logger(func):
    @functools.wraps(func)
    async def inner(*a, **kw):
        print(f"function {func.__name__} start")
        result = await func(*a, **kw)
        print(f"function {func.__name__} end")
        return result

    return inner


# thử log thời gian chạy của 1 hàm chứa limited_fetch_data() chạy 20 task async
@measure_time
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
@functools.lru_cache(maxsize=128)
async def run_limited_fetch():
    await asyncio.gather(*[limited_fetch_data(f"query {i}") for i in range(20)])


# test thử decorator measure_time
await run_limited_fetch()


### IV: Context manager:
# làm sao chắc chắn tài nguyên được giải phóng sau khi dùng xong,
# tránh memory leak, kể cả khi có exception xảy ra

## 1. with và contextlib.contextmanager:
# không với with: dễ quên f.close() → rò rỉ file handle
f = open("test.txt", "w")
data = f.read()  # lỗi ở đây -> rò rỉ vì f.close() không được gọi
f.close()

# với with: tự động, kể cả khi có exception
with open("test.txt", "w") as f:
    data = f.read()

# => khối with đảm bảo SETUP và TEARDOWN được gọi đúng cách, kể cả khi có exception xảy ra
# nên dùng with cho các tài nguyên cần giải phóng như file, socket, database connection, lock, semaphore,...

## 2. tự tạo context manager với contextlib.contextmanager:
import contextlib
import time


@contextlib.contextmanager
def timer(label: str):
    # SETUP (chạy trước khi vào khối with)
    print("Setting up...")
    t = time.perf_counter()
    try:
        yield  # code trong khối with sẽ chạy ở đây
    finally:
        # TEARDOWN (chạy sau khi ra khỏi khối with, kể cả khi có exception)
        print(f"Tearing down... {label}: {time.perf_counter() - t}")


with timer("my block"):
    time.sleep(1)  # code trong khối with sẽ chạy ở đây
    print("Doing something...")

# trong AI, torch.no_grad() là 1 context manager để tắt gradient calculation khi inference, tiết kiệm memory và tăng tốc độ


### 1 pipeline gọi LLM - cả 4 công cụ gặp nhau
# Pydantic: Validate output LLM
# async + semaphore: Batch run with concurrency limit
# decorator: Retry + log + cache
# with + context manager: mượn / trả tài nguyên an toàn, kể cả khi lỗi
