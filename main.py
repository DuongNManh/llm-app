import asyncio
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_KEY: str
    model: str = "gpt-4o-mini"
    # ANTHROPIC_KEY: str # mở command nếu muốn lỗi
    model_config = {"env_file": ".env"}


settings = Settings()  # type: ignore


async def main():

    print(f"Using setting: {settings.model}")


if __name__ == "__main__":
    asyncio.run(main())
