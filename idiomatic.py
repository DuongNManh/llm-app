import functools
import asyncio
import time
import contextlib
import random
from pydantic import BaseModel

list_str = [
    "Câu 1: Giải vô địch bóng đá thế giới (FIFA World Cup) đầu tiên trong lịch sử được tổ chức vào năm nào và ở quốc gia nào?",
    "Câu 2: Đội tuyển quốc gia nào đang giữ kỷ lục giành nhiều chức vô địch World Cup nhất?",
    "Câu 3: Ai là cầu thủ ghi nhiều bàn thắng nhất trong lịch sử các vòng chung kết World Cup?",
    "Câu 4: Quốc gia nào là nhà đương kim vô địch (tính đến thời điểm trước khi bước vào World Cup 2026)?",
    "Câu 5: Trong lịch sử bóng đá, huyền thoại nào là cầu thủ duy nhất từng 3 lần nâng cao cúp vàng World Cup?",
    "Câu 6: Vòng chung kết World Cup 2026 lập kỷ lục khi được đồng đăng cai bởi 3 quốc gia. Đó là những quốc gia nào?",
    "Câu 7: Kể từ World Cup 2026, giải đấu mở rộng quy mô lớn nhất lịch sử với bao nhiêu đội tuyển tham gia tranh tài?",
    "Câu 8: Ở World Cup 1986, Diego Maradona đã ghi một bàn thắng bằng tay gây tranh cãi nhất lịch sử. Bàn thắng này thường được gọi với cái tên nổi tiếng nào?",
    "Câu 9: Một kỳ World Cup đặc biệt diễn ra ở Bắc Mỹ, trải dài qua nhiều quốc gia và thành phố lớn.Hỏi: Quốc gia nào dưới đây KHÔNG phải là nước đồng chủ nhà đăng cai tổ chức Vòng chung kết World Cup 2026?",
    "Câu 10: World Cup 2026 đánh dấu một bước ngoặt lớn trong lịch sử bóng đá thế giới về số lượng đội bóng tham dự. Số lượng đội tuyển quốc gia tham dự vòng chung kết FIFA World Cup 2026 đã được mở rộng lên con số bao nhiêu?",
]


class Summary(BaseModel):
    word_count: int
    text: str


_cache: dict[str, Summary] = {}


@contextlib.contextmanager
def measure_time():
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"Time taken: {end - start}")


def retry(times: int):
    def decorator(func):
        @functools.wraps(func)
        async def inner(*a, **kw):
            attemp = 0
            while attemp in range(times):
                try:
                    return await func(*a, **kw)
                except Exception as e:
                    print(f"Attemp {attemp + 1} failed: {e}")
                    attemp += 1
                    if attemp == times:
                        raise
                    await asyncio.sleep(1)

        return inner

    return decorator


def logger(func):
    @functools.wraps(func)
    async def inner(*a, **kw):
        try:
            result = await func(*a, **kw)
            return result
        except Exception as e:
            print(f"Exception occur {func.__name__} : {e}")
            raise

    return inner


async def summarize(text: str) -> Summary:
    if text in _cache:
        return _cache[text]
    random_ex = random.choice([True, False, False])
    if random_ex:
        raise Exception(f"llm exception occur for {text[:10]}...")
    await asyncio.sleep(1)
    word_count = len(text)
    content = "Summary: " + text
    print(f"Summarizing: {text[:10]}...{word_count} words")
    _cache[text] = Summary(word_count=word_count, text=content)
    return Summary(word_count=word_count, text=content)


semaphore = asyncio.Semaphore(3)


# @retry(times=3)
async def rate_limited_summarize(text: str) -> Summary:
    async with semaphore:
        return await summarize(text)


@logger
@retry(times=3)
async def call_llm(doc: list[str]) -> list[Summary]:
    return await asyncio.gather(*[rate_limited_summarize(d) for d in doc])
