"""bài tập cho lý thuyết module 03-gọi api bất đồng bộ, retry, limiter, semaphore"""

import time
import asyncio
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception,
)
import httpx
from aiolimiter import AsyncLimiter


class Post(BaseModel):
    """Model for a post from JSONPlaceholder API"""

    user_id: int = Field(alias="userId")
    id: int
    title: str
    body: str | None = "Body not provided"


# timeout for httpx client
timeout = httpx.Timeout(read=10, write=5, pool=3, connect=3)
# limit cho httpx client
limits = httpx.Limits(
    max_keepalive_connections=5, max_connections=10, keepalive_expiry=5
)

# semaphore giới hạn số lượng task chạy song song là 10
semaphore = asyncio.Semaphore(10)
# limiter tránh việc gửi burst request API quá nhiều, dẫn đến server bị quá tải hoặc bị chặn (429).
limiter = AsyncLimiter(max_rate=5, time_period=1)


def should_retry(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True

    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600 or exc.response.status_code == 429

    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(),
    retry=retry_if_exception(should_retry),
)
async def fetch_post_v2(client: httpx.AsyncClient, url: str) -> Post:
    try:
        response = await client.get(url)
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
            print(f"Server error while fetching post from {url}: {e}")
        if 400 <= code < 500:
            print(f"Client error while fetching post from {url}: {e}")
        raise e


async def fetch_posts_with_sem_lim(client: httpx.AsyncClient, url: str) -> Post:
    async with semaphore, limiter:  # chú ý semaphore và limiter là global
        post = await fetch_post_v2(client, url)
        return post


async def main_v2():
    start = time.perf_counter()
    urls = [f"/posts/{i}" for i in range(-5, 101)]
    async with httpx.AsyncClient(
        timeout=timeout, limits=limits, base_url="https://jsonplaceholder.typicode.com"
    ) as client:
        tasks = [fetch_posts_with_sem_lim(client, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        ok = [r for r in results if isinstance(r, Post)]
        err = [r for r in results if isinstance(r, Exception)]

        print(f"Total: {len(results)}, OK {len(ok)} / lỗi {len(err)}")

        with open("out.json", "w", encoding="utf-8") as f:
            f.write("[\n")
            for post in ok:
                f.write(post.model_dump_json() + ",\n")
            f.write("]\n")

    end = time.perf_counter()
    print(f"Total time: {end - start:.2f} seconds")


asyncio.run(main_v2())

# await main_v2()
