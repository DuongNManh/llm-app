### Gọi API Async với httpx, asyncio, retry, limiter, semaphore

# mục tiêu: gọi API hàng loạt nhanh, bền, rồi response về dạng model sạch
# kết hợp các kỹ thuật: async, retry, limiter, semaphore, cache, log, đo thời gian chạy
# response -> model sạch: json -> pydantic model, validation, phân trang, streaming
# pipeline: fetch -> validate -> transform -> gom, xử lý lỗi từng phần

# Vòng đời một lời gọi
# gửi request (httpx AsyncClient)
# → chờ mạng (nên async cùng semaphore (task song song trong app), limiter (rate limit API))
# → nhận response + status
# → lỗi? retry / bỏ qua (@retry)
# → parse JSON + validate (pydantic)
# → dữ liệu sạch để dùng

## httpx AsyncClient:

# tài liệu: https://www.python-httpx.org/async/
# 1. httpx là HTTP client hiện đại, hỗ trợ cả sync lẫn async, API giống 'requests'
# quen thuộc nhưng có await. AsyncClient là bản bất đồng bộ.


# 2. một client dùng chung cho nhiều request
import httpx


async def demo_httpx():
    urls = ["step1", "step2", "step3"]
    # ✗ SAI: client trong vòng lặp (tạo lại client mỗi lần gọi API)
    for u in urls:
        async with httpx.AsyncClient() as c:
            await c.get(u)  # mở lại kết nối mỗi lần

        # ✓ ĐÚNG: một client cho tất cả
        async with httpx.AsyncClient() as c:
            for u in urls:
                await c.get(u)


# -> tận dụng connection pooling, giảm overhead, tăng tốc độ


# 3. timeout, connection pool
# đây là 2 setup dành cho AsyncClient, tránh timeout, quá tải kết nối

timeout = httpx.Timeout(
    connect=3, read=10, write=5, pool=3
)  # timeout for httpx client, read 10s, write 5s, pool 3s (tối đa 3s để lấy connection từ pool), connect 3s (tối đa 3s để kết nối đến server)
limits = httpx.Limits(
    max_connections=10, max_keepalive_connections=5
)  # limit for httpx client,
# `httpx.Limits` được sử dụng để giới hạn số lượng kết nối mà client có thể mở cùng một lúc. Cụ thể:
# - `max_keepalive_connections=5`: Giới hạn số lượng kết nối giữ sống (keep-alive) mà client có thể duy trì. Khi một kết nối được giữ sống, nó có thể được tái sử dụng cho các yêu cầu tiếp theo đến cùng một máy chủ, giúp giảm độ trễ và tăng hiệu suất.
# - `max_connections=10`: Giới hạn tổng số kết nối mà client có thể mở cùng một lúc, bao gồm cả kết nối giữ sống và kết nối mới. Khi số lượng kết nối đạt đến giới hạn này, các yêu cầu mới sẽ phải chờ cho đến khi có kết nối được giải phóng.


# kết hợp tất cả các thiết lập trên để tạo một AsyncClient dùng chung cho tất cả các request, với timeout và connection pool hợp lý, giúp tăng tốc độ và độ bền khi gọi API hàng loạt.
async def main():
    async with httpx.AsyncClient(
        timeout=timeout, limits=limits, base_url="https://jsonplaceholder.typicode.com"
    ) as client:
        for id in range(1, 10):
            response = await client.get(f"/posts/{id}")
            response.raise_for_status()
            post = response.json()
            print(post)


## gọi API bền vững

# 1. bắt và phân loại lỗi
# raise_for_status()
# httpx ko tự báo error khi status 4xx/5xx. Gọi raise_for_status() để biến lỗi status thành exception mà ta có thể bắt và xử lý.
# phân loại lỗi: 4xx (client error) -> bỏ qua, 5xx (server error) -> retry; timeout -> retry
# async def demo_error_handling(c, url):
#     try:
#     r = await c.get(url)
#     r.raise_for_status() # 4xx/5xx -> raise
#     except httpx.TimeoutException:
#     ... # tạm thời -> retry
#     except httpx.HTTPStatusError as e:
#     code = e.response.status_code
#     ... # 4xx: bỏ; 5xx: retry

# 2. retry + exponential backoff + jitter
# retry: trên các lỗi tạm thời (đã phân loại ở trên), retry với số lần giới hạn, delay tăng dần (exponential backoff) + random jitter để tránh đồng loạt retry
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(),
    retry=retry_if_exception_type(httpx.TransportError),
)
async def fetch(c, url): ...


# 3. semaphore + limiter
# semaphore: giới hạn số lượng request đang bay cùng lúc. Bảo vệ máy bạn và pool kết nối
# limiter: giới hạn tốc độ request/s (khác với song song). 10 request song song nhưng bị 5 r/s giới hạn, tránh client bị chặn (429).
# 2 kỹ thuật này thường cùng dùng chung để vừa bảo vệ client, vừa bảo vệ server, vừa tránh bị chặn.
# số song song
import asyncio
from aiolimiter import AsyncLimiter

sem = asyncio.Semaphore(10)
# tốc độ: 5 request mỗi giây
lim = AsyncLimiter(5, 1)


async def fetch_with_semaphore_limiter(c, url):
    async with sem, lim:
        await fetch(c, url)


## kết hợp tất cả các kỹ thuật trên để gọi API hàng loạt bền vững, nhanh chóng, và nhận về dữ liệu sạch.
def should_retry(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True

    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 429 or 500 <= code < 600:  # retry on 429 or 5xx
            return True

    return False


# cả sem và lim là global
sem = asyncio.Semaphore(10)  # giới hạn số request song song
lim = AsyncLimiter(max_rate=5, time_period=1)  # giới hạn tốc độ r/s


# đảm nhận việc semaphore + limiter cho request
async def fetch_with_sem_lim(c, url):
    urls = [url]  # giả sử url là một danh sách các url cần fetch
    async with httpx.AsyncClient(
        timeout=timeout, limits=limits, base_url="https://jsonplaceholder.typicode.com"
    ) as client:
        async with sem, lim:
            return await fetch(c, url)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(),
    retry=retry_if_exception(should_retry),
)
async def fetch_posts_v1(client: httpx.AsyncClient, url: str):
    try:
        response = await client.get(url)
        response.raise_for_status()  # biến lỗi status 4xx/5xx thành exception để xử lý
        post = response.json()
        print(f"Fetched post {id}: {post}")
    except httpx.TimeoutException as e:
        # retry the request if a timeout occurs
        print(f"Timeout occurred for post {id}: {e}")
        raise e
    except httpx.HTTPStatusError as e:
        print(f"Request error occurred for post {id}: {e}")
        code = e.response.status_code
        if 500 <= code < 600:
            # retry the request if a server error occurs
            print(f"Server error occurred for post {id}: {e}")
        if 400 <= code < 500:
            print(f"Client error occurred for post {id}: {e}")
        raise e


# tới đây, ta đã có thể gọi api hàng loạt và bền vững với 1 client dùng chung, và các kỹ thuật semaphore, limiter, retry, error handling, timeout, connection pool. Dữ liệu trả về có thể parse và validate bằng pydantic model để đảm bảo sạch sẽ và đúng định dạng.


### Response -> dữ liệu sạch

# biến json thô (có thể thiếu/sai/nhiều trang) thành object đáng tin

# 1. json -> pydantic (validate response)
# vấn đề: LLM hoặc API trả về json r.json() trần. Dễ sai field, thiếu field, sai kiểu mà không biết ta cần 1 cổng để kiểm tra (validate).
# giải pháp: định nghĩa model (pydantic) và dùng Model.model_validate(json) để validate. Nếu thiếu field hoặc sai kiểu, sẽ raise ValidationError. Nếu đúng, trả về model sạch.

from pydantic import BaseModel, ValidationError


class Post(BaseModel):
    userId: int
    id: int
    title: str
    body: str


data = r.json()  # giả sử đây là json thô từ API
post = Post.model_validate(data)  # validate và parse json thành model sạch

# 2. dữ liệu "bẩn" ngoài đời thực
# field có khi thiếu (dùng giá trị default), có khi null (X | None), tên field theo camelCase
# pydantic hỗ trợ đặt default cho field, hỗ trợ X | Null (Optional) và Alias user_id: int (Field(alias="camelCase")) để map tên field khác nhau giữa json và model.

from pydantic import BaseModel, Field


class User(BaseModel):
    user_id: int = Field(alias="userId")
    name: str
    email: str | None = None
    tags: list[str] = []


User.model_validate(data)  # map + default

# 3. phân trang & streaming
# API trả về nhiều trang (pagination) hoặc stream dữ liệu (streaming LLM). Ta cần

# phân trang: gọi API nhiều lần với page=1,2,3,... cho đến khi hết dữ liệu. Mỗi lần gọi API, validate json -> model sạch.


# phân trang: lặp tới khi hết dữ liệu
async def fetch_paginated(c, url):
    async with httpx.AsyncClient(
        timeout=timeout, limits=limits, base_url="https://jsonplaceholder.typicode.com"
    ) as c:
        page = 1
        has_more = True
        while has_more:
            r = await c.get(url, params={"page": page})
            r.raise_for_status()
            data = r.json()
            has_more = data.get("next", False)
            page += 1


# streaming: đọc từng chunk dữ liệu từ response.iter_bytes() hoặc response.iter_lines(), validate từng chunk -> model sạch.
import json


async def fetch_streaming(c, url):
    async with httpx.AsyncClient(
        timeout=timeout, limits=limits, base_url="https://jsonplaceholder.typicode.com"
    ) as c:
        async with c.stream("GET", url) as response:
            async for chunk in response.aiter_bytes():
                data = json.loads(chunk)
                model = BaseModel.model_validate(data)
                return model  # trả về model sạch từng chunk hoặc gom tất cả model vào list để trả về
            async for line in response.aiter_lines():
                data = json.loads(line)
                model = BaseModel.model_validate(data)
                return model  # trả về model sạch cuối cùng hoặc gom tất cả model vào list để trả về


# thử kết hợp cho streaming LLM và các kĩ thuật ở trên (AsyncClient, semaphore, limiter, retry, validate) để gọi API LLM hàng loạt bền vững và nhận về dữ liệu sạch.


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(),
    retry=retry_if_exception(should_retry),
)
async def fetch_stream(http_client, url):
    async with http_client.stream("GET", url) as response:
        async for line in response.aiter_lines():
            data = json.loads(line)
            model = BaseModel.model_validate(data)
            return model


async def fetch_llm_stream(c, url):
    async with httpx.AsyncClient(
        timeout=timeout, limits=limits, base_url="https://jsonplaceholder.typicode.com"
    ) as client:
        async with sem, lim:
            await fetch_stream(client, url)


### gom, đếm và ghi kết quả

# 1. sau khi gọi API hàng loạt, gom kết quả về 1 list, đếm số lượng thành công/thất bại, ghi log hoặc lưu vào file. không nên raise exception nếu 1 request thất bại, mà nên gom tất cả kết quả về và xử lý từng phần.

# return_exceptions=True: gather hết. danh sách có cả kết quả và exception.
# Sau đó ta tách, đếm và gom thành 2 list: thành công và thất bại. Ghi log hoặc lưu vào file để kiểm tra.


async def process_results(results):
    import json

    results = await asyncio.gather(
        *[fetch_llm_stream(c, url) for url in urls], return_exceptions=True
    )

    ok = [r for r in results if not isinstance(r, Exception)]
    err = [r for r in results if isinstance(r, Exception)]

    with open("results.json", "w") as f:
        json.dump([result.dict() for result in ok], f)


### pipeline async hoàn chỉnh:

# 1 AsyncClient dùng chung -> 2. semaphore + limiter (bọc ngoài http call) - > 3. retry + exponential backoff + jitter (bọc ngay trên http call) -> 4. validate json -> model sạch -> 5. gom kết quả, đếm, log, lưu file

# ta cùng làm 1 ví dụ pipeline async hoàn chỉnh: gọi API hàng loạt, bền vững, nhận về dữ liệu sạch, gom kết quả, đếm, log, lưu file.

import asyncio, httpx, json
from pydantic import BaseModel, ValidationError, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception,
)
from aiolimiter import AsyncLimiter


class Post(BaseModel):
    user_id: int = Field(alias="userId")
    id: int
    title: str
    body: str | None = "Body not provided"


# với hàm gọi httpx, ta dùng retry khi http call gặp lỗi, ví dụ timeout, server error 5xx.
# vì sao không dùng semaphore và limiter trong hàm fetch_post_final (httpx call) mà lại dùng trong hàm fetch_posts_with_sem_lim (gọi nhiều httpx call)?
# trong hàm fetch_post_final, ta chỉ gọi 1 httpx call, nên không cần semaphore và limiter
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(),
    retry=retry_if_exception(should_retry),
)
async def fetch_post_final(c: httpx.AsyncClient, url: str) -> Post:
    try:
        response = await c.get(url)
        response.raise_for_status()
        post = Post.model_validate(response.json())
        print(f"Fetched post from {url}: {post}")
        return post
    except httpx.TimeoutException as e:
        print(f"Timeout occurred while fetching post from {url}: {e}")
        raise e
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        if 500 <= code < 600:
            print(f"Client Error for {url}, retry...")
        if 400 <= code < 429:
            print(f"Request Error for {url}")
        raise e


# global semaphore + limiter để bọc http call
sem = asyncio.Semaphore(10)  # giới hạn số request song song
limiter = AsyncLimiter(
    max_rate=5, time_period=1
)  # giới hạn tốc độ r/s # tránh việc gửi burst request API quá nhiều, dẫn đến server bị quá tải hoặc bị chặn (429).
# dù ta có 10 semaphore, nhưng vẫn chỉ được gửi tối đa 5 request/giây, vì limiter là global, nên các request sẽ phải chờ nhau để đảm bảo giới hạn tốc độ request.


# sử dụng global semaphore + limiter để bọc http call
# khi main gọi tasks = [client.get(url) for url in urls], thì các request được gửi đi cùng lúc, nên cần semaphore và limiter
# nhưng vì sao chỉ có 1 url ở param, nhưng vẫn cần semaphore và limiter trong hàm fetch_with_sem_lim_final thay vì là hàm main?
# nếu ta để semaphore và limiter trong hàm main bọc ngoài tasks, thì thực tế ta chỉ dùng 1 semaphore và limiter cho toàn bộ các request song song
# nhưng nếu ta để semaphore và limiter trong hàm fetch_with_sem_lim_final, thì mỗi request sẽ có 1 semaphore và limiter riêng, nên sẽ giới hạn
# vậy vì sao vẫn biết được số semaphore và limiter còn lại dù ta chỉ còn để trong mỗi request, vì semaphore và limiter là global, nên khi 1 request lấy semaphore và limiter, thì các request khác sẽ phải chờ, nên vẫn giới hạn được số request song song, và vẫn đảm bảo giới hạn tốc độ request.
async def fetch_with_sem_lim_final(c, url) -> Post:
    async with sem, limiter:
        results = await fetch_post_final(c, url)
        return results


timeout = httpx.Timeout(
    read=10, write=5, pool=3, connect=3
)  # timeout for httpx client, read 10s, write 5s, pool 3s (tối đa 3s để lấy connection từ pool), connect 3s (tối đa 3s để kết nối đến server)

limits = httpx.Limits(
    max_keepalive_connections=5, max_connections=10, keepalive_expiry=5
)  # limit for httpx client,
# `httpx.Limits` được sử dụng để giới hạn số lượng kết nối mà client có thể mở cùng một lúc. Cụ thể:
# - `max_keepalive_connections=5`: Giới hạn số lượng kết nối giữ sống (keep-alive) mà client có thể duy trì. Khi một kết nối được giữ sống, nó có thể được tái sử dụng cho các yêu cầu tiếp theo đến cùng một máy chủ, giúp giảm độ trễ và tăng hiệu suất.
# - `max_connections=10`: Giới hạn tổng số kết nối mà client có thể mở cùng một lúc, bao gồm cả kết nối giữ sống và kết nối mới. Khi số lượng kết nối đạt đến giới hạn này, các yêu cầu mới sẽ phải chờ cho đến khi có kết nối được giải phóng.


async def main_pipeline():
    # tạo 1 AsyncClient dùng chung cho tất cả các request, với thiết lập timeout và connection pool hợp lý
    async with httpx.AsyncClient(
        timeout=timeout, limits=limits, base_url="https://jsonplaceholder.typicode.com"
    ) as client:
        urls = [f"/posts/{i}" for i in range(-10, 10)]  # giả sử có các url lỗi
        tasks = [fetch_with_sem_lim_final(client, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        ok = [
            r for r in results if isinstance(r, Post) and not isinstance(r, Exception)
        ]
        err = [r for r in results if isinstance(r, Exception)]

        print(f"Total: {len(results)}, Ok {len(ok)}, Lỗi {len(err)}")

        with open("results.json", "w", encoding="utf-8") as f:
            json.dump(
                [result.model_dump() for result in ok], f, ensure_ascii=False, indent=2
            )


### ví dụ các lỗi kinh điển


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(),
    retry=retry_if_exception(should_retry),
)
async def fetch_posts(ids: list[int]):
    # lỗi kinh điển khi dùng httpx.AsyncClient() trong hàm fetch_posts, vì mỗi lần gọi hàm fetch_posts, ta lại tạo ra 1 httpx.AsyncClient() mới, nên sẽ không tận dụng được connection
    async with httpx.AsyncClient(
        timeout=timeout, limits=limits, base_url="https://jsonplaceholder.typicode.com"
    ) as client:
        async with sem, lim:
            for id in ids:
                try:
                    response = await client.get(f"/posts/{id}")
                    response.raise_for_status()
                    post = Post.model_validate(response.json())
                    print(f"Fetched post {id}: {post}")
                    return post
                except httpx.TimeoutException as e:
                    # retry the request if a timeout occurs
                    print(f"Timeout occurred for post {id}: {e}")
                    raise e
                except httpx.HTTPStatusError as e:
                    code = e.response.status_code
                    print(f"Request status {code} for post {id}: {e}")
                    raise e


async def main():
    async with (
        sem,
        lim,
    ):  # lỗi khi dùng semaphore và limiter ở đây, vì ta chỉ dùng 1 semaphore và limiter cho toàn bộ các request song song, nên sẽ không giới hạn được số request song song, và cũng không đảm bảo giới hạn tốc độ request.
        tasks = [fetch_posts([i]) for i in range(-5, 10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for post in results:
            print(post)

        ok = [r for r in results if isinstance(r, Post)]
        err = [r for r in results if isinstance(r, Exception)]

        print(f"OK {len(ok)} / lỗi {len(err)}")

        with open("out.json", "w", encoding="utf-8") as f:
            f.write("[\n")
            for post in ok:
                f.write(post.model_dump_json() + ",\n")
            f.write("]\n")


# await main()
