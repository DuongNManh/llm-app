### Function calling và tool calling trong LLMs và Python

# 1. Function calling
# - a mechanism that allows a LLM to call external functions or APIs during its excecution and receive structured responses (JSON,etc.) from those functions. It enables the model to interact with external systems, retrieve data, or perform specific actions based on the context of the conversation. It can be your own function in python
# for example:
# User: What is the price of My Quang?
# Model: tool_call: get_product_info(product_name="My Quang")
# Code: get_product_info(product_name: str) -> dict:
#     # logic to fetch product info from a database or API
# Code: return {"product_name": product_name, "price": 10.99, "availability": "In Stock"}
# Model: The price of My Quang is $10.99 and it is currently in stock.

# Block "Code" needed developer to implement and create a way for the model to connect to the function and call it.

from google.genai import Client
from pydantic_settings import BaseSettings
import tiktoken
from google.genai import types


class Settings(BaseSettings):
    GEMINI_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    model_config = {"env_file": ".env"}


settings = Settings()
encoder = tiktoken.get_encoding("o200k_base")

prompt = """
You are a helpful product assitant for a restaurant.
Customer question: What;s the current price for My Quang and how much does it cost for a 10 bowls order?
Response with the product information and the total cost for 10 bowls.
"""

PRODUCT_INFO = {
    "my_quang": {
        "product_name": "My Quang",
        "price": 10.99,
        "stocks": "10",
        "unit": "bowl",
    },
    "pho": {"product_name": "Pho", "price": 8.99, "stocks": "3", "unit": "bowl"},
    "banh_mi": {
        "product_name": "Banh Mi",
        "price": 5.49,
        "stocks": "0",
        "unit": "piece",
    },
    "bun Cha": {
        "product_name": "Bun Cha",
        "price": 9.99,
        "stocks": "15",
        "unit": "bowl",
    },
    "goi_cuon": {
        "product_name": "Goi Cuon",
        "price": 6.99,
        "stocks": "5",
        "unit": "piece",
    },
    "banh_xeo": {
        "product_name": "Banh Xeo",
        "price": 7.99,
        "stocks": "0",
        "unit": "piece",
    },
    "bun_bo_hue": {
        "product_name": "Bun Bo Hue",
        "price": 11.49,
        "stocks": "7",
        "unit": "bowl",
    },
    "com_tam": {
        "product_name": "Com Tam",
        "price": 8.49,
        "stocks": "2",
        "unit": "bowl",
    },
    "banh_bao": {
        "product_name": "Banh Bao",
        "price": 4.99,
        "stocks": "5",
        "unit": "piece",
    },
    "banh_cuon": {
        "product_name": "Banh Cuon",
        "price": 6.49,
        "stocks": "0",
        "unit": "piece",
    },
}


def get_product_info(product_name: str) -> dict:
    """Simulate fetching product info from a database or API"""
    if product_name not in PRODUCT_INFO:
        return {"error": f"Product '{product_name}' not found."}
    return PRODUCT_INFO[product_name]


# LLM can never access the function get_product_info directly, so we need to call the function in our code and return the result to the model.


# if we call the llm like this, 100% we will get a hallucination, because the model doesn't know the price of My Quang, and it will make up a price in our restaurant. So we need to call a function to get the price of My Quang, and then return the price to the model, so the model can calculate the total cost for 10 bowls.
def conversation_with_gemini(prompt: str):
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
    client = Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=150,
            response_mime_type="application/json",
        ),
    )
    return response.text


def main():
    response = conversation_with_gemini(prompt)
    print("Response from Gemini API:")
    print(response)


if __name__ == "__main__":
    main()
