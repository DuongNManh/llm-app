from pydantic_settings import BaseSettings
from google import genai
from openai import OpenAI
from anthropic import Anthropic

cache = dict()


class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    model_config = {"env_file": ".env"}


settings = Settings()


def call_gemini_api(prompt: str):
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
    if (prompt, "gemini") in cache:
        return cache[(prompt, "gemini")]
    client_gemini = genai.Client(api_key=settings.GEMINI_API_KEY)
    resp = client_gemini.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
    )
    cache[(prompt, "gemini")] = resp.text
    return resp.text


def call_openai_api(prompt: str):
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
    if (prompt, "openai") in cache:
        return cache[(prompt, "openai")]
    client_openai = OpenAI(api_key=settings.OPENAI_API_KEY)
    resp = client_openai.responses.create(
        model="gpt-4o-mini", input=prompt, max_output_tokens=500
    )

    cache[(prompt, "openai")] = resp.output_text
    return resp.output_text


def call_openrouter_api(prompt: str):
    api_key = settings.OPENROUTER_API_KEY
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.")
    if (prompt, "openrouter") in cache:
        return cache[(prompt, "openrouter")]
    client_openrouter = OpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=settings.OPENROUTER_API_KEY
    )
    resp = client_openrouter.chat.completions.create(
        model="tencent/hy3:free",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    cache[(prompt, "openrouter")] = resp.choices[0].message.content
    return resp.choices[0].message.content


def call_anthropic_api(prompt: str):
    api_key = settings.ANTHROPIC_API_KEY
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set in the environment variables.")
    if (prompt, "anthropic") in cache:
        return cache[(prompt, "anthropic")]
    client_anthropic = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    resp = client_anthropic.messages.create(
        model="claude-sonnet-4.5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    cache[(prompt, "anthropic")] = resp.content
    return resp.content


PROVIDERS = {
    "openai": call_openai_api,
    "gemini": call_gemini_api,
    "claude": call_anthropic_api,
    "openrouter": call_openrouter_api,
}


def ask_llm(provider: str, prompt: str):
    if prompt is None:
        raise ValueError("Prompt cannot be None.")
    fn = PROVIDERS.get(provider)
    if not fn:
        raise ValueError(f"Unknown provider: {provider}")
    return fn(prompt)


def main():
    prompt = input("Enter your prompt: ")
    provider = ["hihi", "openai", "gemini", "claude", "openrouter"]

    for p in provider:
        try:
            result = ask_llm(p, prompt)
            print(f"{p} response: {result}")
        except Exception as e:
            print(f"Error with provider {p}: {e}")


if __name__ == "__main__":
    main()
