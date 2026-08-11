# bài tập sử dụng tool calling, agent loop cho product assistant, có 2 tool là get product info và calculate item cost.

from decimal import Decimal
import json
from typing import cast
from openai import Client
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from openai.types.chat import ChatCompletionToolUnionParam
import tiktoken


class Settings(BaseSettings):
    GEMINI_API_KEY: str | None = None
    OPENROUTER_API_KEY_2: str | None = None
    model_config = {"env_file": ".env"}


settings = Settings()
encoder = tiktoken.get_encoding("o200k_base")

PRODUCT_INFO = {
    "my_quang": {
        "product_name": "My Quang",
        "price": "10.99",
        "stocks": "10",
        "unit": "bowl",
    },
    "pho": {"product_name": "Pho", "price": "8.99", "stocks": "7", "unit": "bowl"},
    "banh_mi": {
        "product_name": "Banh Mi",
        "price": "5.49",
        "stocks": "20",
        "unit": "piece",
    },
    "bun_cha": {
        "product_name": "Bun Cha",
        "price": "9.99",
        "stocks": "8",
        "unit": "bowl",
    },
    "goi_cuon": {
        "product_name": "Goi Cuon",
        "price": "6.99",
        "stocks": "6",
        "unit": "piece",
    },
    "banh_xeo": {
        "product_name": "Banh Xeo",
        "price": "7.99",
        "stocks": "15",
        "unit": "piece",
    },
    "bun_bo_hue": {
        "product_name": "Bun Bo Hue",
        "price": "11.49",
        "stocks": "7",
        "unit": "bowl",
    },
    "com_tam": {
        "product_name": "Com Tam",
        "price": "8.49",
        "stocks": "2",
        "unit": "bowl",
    },
    "banh_bao": {
        "product_name": "Banh Bao",
        "price": "4.99",
        "stocks": "5",
        "unit": "piece",
    },
    "banh_cuon": {
        "product_name": "Banh Cuon",
        "price": "6.49",
        "stocks": "0",
        "unit": "piece",
    },
}


class ProductNotFoundError(Exception):
    """Custom exception for product not found errors."""

    pass


class QuantityError(Exception):
    """Custom exception for invalid quantity errors."""

    pass


class GetProductInfoParams(BaseModel):
    """Schema for the get_product_info tool parameters."""

    product_id: str = Field(
        ...,
        description="The unique identifier of the product. For Example: 'my_quang', 'banh_chung', 'pho', etc.",
    )


class ProductInfoResult(BaseModel):
    """Schema for the result of the get_product_info tool."""

    product_name: str = Field(..., description="The name of the product.")
    price: str
    stocks: str
    unit: str


def get_product_info(product_id: str) -> ProductInfoResult:
    """Simulate fetching product info from a database or API"""
    if product_id not in PRODUCT_INFO:
        raise ProductNotFoundError(f"Product '{product_id}' not found.")
    return ProductInfoResult(**PRODUCT_INFO[product_id])


class ItemCostResult(BaseModel):
    """Schema for the result of the calculate_item_cost tool."""

    product_name: str
    quantity: int
    unit_price: str
    total_price: str
    discount: str


class CalculateItemCostParams(BaseModel):
    """Schema for the calculate_item_cost tool parameters."""

    product_id: str = Field(
        ...,
        description="The unique identifier of the product. For Example: 'my_quang', 'banh_chung', 'pho', etc.",
    )
    quantity: int = Field(
        ...,
        description="The quantity of the product to calculate the total cost for. For example, '2', '5', '10', etc.",
    )


def calculate_item_cost(product_id: str, quantity: int) -> ItemCostResult:
    """Calculate total cost based on product ID and quantity"""
    product_info = get_product_info(product_id)
    if quantity <= 0:
        raise QuantityError("You must order at least one item.")

    base_price = Decimal(product_info.price)
    discount = Decimal(0)
    if quantity >= 5:
        discount = Decimal(0.1)  # 10% discount for orders of 5 or more
    if quantity >= 10:
        discount = Decimal(0.25)  # 25% discount for orders of 10 or more

    unit_price = base_price * (Decimal(1) - discount)
    total_price = unit_price * quantity

    unit_price = unit_price.quantize(Decimal("0.01"))  # Round to 2 decimal places
    total_price = total_price.quantize(Decimal("0.01"))  # Round to 2 decimal places
    discount = (discount * 100).quantize(Decimal("0.01"))
    return ItemCostResult(
        product_name=product_info.product_name,
        quantity=quantity,
        unit_price=str(unit_price),
        total_price=str(total_price),
        discount=str(discount),
    )


READ_TOOLS = {
    "calculate_item_cost": calculate_item_cost,
    "get_product_info": get_product_info,
    # "calculate_bill": calculate_bill,
}

WRITE_TOOLS = {
    # "payment_order": payment_order,
}

if not settings.OPENROUTER_API_KEY_2:
    raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.")

client = Client(
    api_key=settings.OPENROUTER_API_KEY_2, base_url="https://openrouter.ai/api/v1"
)


class ToolError(BaseModel):
    """Schema for error responses from tools."""

    error: bool = Field(..., description="Indicates if an error occurred.")
    message: str = Field(..., description="A message describing the error.")


tools = [
    {
        # tool 1: get product info in restaurant's menu
        "type": "function",
        "function": {
            "name": "get_product_info",
            "description": "Retrieve information about a product, including its price and availability.",
            "strict": True,
            "parameters": GetProductInfoParams.model_json_schema(),
        },
    },
    {
        # tool 4: calculate the total cost for a specific item based on quantity
        "type": "function",
        "function": {
            "name": "calculate_item_cost",
            "description": "Calculate the total cost for a specific item based on quantity.",
            "strict": True,
            "parameters": CalculateItemCostParams.model_json_schema(),
        },
    },
]


def build_messages(input_prompt: str) -> list[dict]:
    """Build the initial messages for the chat completion request."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful product assitant for a restaurant. Always use the available tools to look up current product information (parallel calls) and calculate bills (parallel calls) and help users with their orders (place more or finish). Do not rely on general knowledge or make up information or calculate prices manually. Answer in friendly conversational style.",
        },
        {"role": "user", "content": f"{input_prompt}"},
    ]
    return messages


def agent_loop():
    """Main loop for the product assistant agent."""
    place_order_init = input(
        "Enter your order request (e.g., 'I want to order 2 bowls of My Quang and 1 Banh Mi'): "
    )
    messages = build_messages(place_order_init)
    print("Initial messages:")
    print(messages)
    print("================================")
    while True:
        print("Start a new iteration of the agent loop...")
        response = client.chat.completions.create(
            model="nvidia/nemotron-3-super-120b-a12b:free",
            messages=messages,
            tools=cast(list[ChatCompletionToolUnionParam], tools),
            tool_choice="auto",
            temperature=0.2,
            max_tokens=1500,
        )

        print("Response from OpenRouter API:")
        print(response)

        assistance_message = response.choices[0].message
        print(f"Assistant message: {assistance_message}")
        messages.append(assistance_message)

        tool_calls = assistance_message.tool_calls
        print(f"Tool calls: {tool_calls}")

        if not tool_calls:
            print("No tool calls were made in the response.")
            print(response.choices[0].message.content)
            break

        for i, tool_call in enumerate(tool_calls):
            print(f"-------tool call {i + 1}-------\n")
            tool_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"function: {tool_name}\n")
            print(f"arguments: {args}\n")

            try:
                if tool_name in WRITE_TOOLS:
                    print(f"Calling write tool: {tool_name}")
                    confirmation = input(
                        f"Do you want to proceed with calling the write tool '{tool_name}'? (yes/no): "
                    )
                    if confirmation.lower() != "yes":
                        print(
                            f"Skipping the write tool '{tool_name}' as per user confirmation."
                        )
                        result = ToolError(
                            error=True,
                            message=f"User skipped the write tool '{tool_name}'.",
                        )
                    continue
                else:
                    result = READ_TOOLS[tool_name](**args)
            except Exception as e:
                print(f"Error occurred while calling {tool_name}: {e}")
                result = ToolError(error=True, message=str(e))
                continue

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result.model_dump_json()
                    if isinstance(result, BaseModel)
                    else json.dumps(result),
                }
            )


def main():
    agent_loop()


if __name__ == "__main__":
    main()


# WRITE_TOOLS = {
# "payment_order": payment_order,
# }

# class CalculateBillParams(BaseModel):
#     """Schema for an item in the bill, including its cost and discount."""
#     items: list[ItemCostResult] = Field(..., description="List of items in the bill with their costs and discounts.")

# class BillResult(BaseModel):
#     """Schema for the result of the calculate_bill tool."""
#     total_bill_amount: str = Field(..., description="The total bill amount as a string.")
#     items: list[ItemCostResult] = Field(..., description="List of items in the bill with their costs and discounts.")

# def calculate_bill(items: CalculateBillParams) -> BillResult:
#     """Calculate the total bill for an order"""
#     total_bill = Decimal(0)
#     detailed_items = []
#     for item in items.items:
#         total_bill += Decimal(item.total_price)
#         detailed_items.append(item)

#     total_bill = total_bill.quantize(Decimal('0.01'))  # Round to 2 decimal places
#     return BillResult(
#         total_bill_amount=str(total_bill),
#         items=detailed_items
#     )

# class SubmitOrderResult(BaseModel):
#     """Schema for the result of the payment_order tool."""
#     order_id: str
#     status: str
#     created_at: str

# class OrderSubmitParams(BaseModel):
#     """Schema for the parameters required to submit an order."""
#     order_id: str = Field(..., description="The unique identifier for the order.")
#     bill_result: BillResult = Field(..., description="The result of the bill calculation, including items and total amount.")

# # write function, tránh cho llm tự động call, cần xác nhận trước khi call.
# def payment_order(order_params: OrderSubmitParams) -> SubmitOrderResult:
#     """Simulate payment the order request and return the result"""
#     # This function can be expanded to handle order submission logic
#     print("======Payment order with the following details:======")

#     for item in order_params.bill_result.items:
#         print(f"Item: {item.product_name}, Quantity: {item.quantity}, Total Price: {item.total_price}")

#     return SubmitOrderResult(
#         order_id=order_params.order_id,
#         status="payment_successful",
#         created_at="2023-01-01T00:00:00Z"
#     )


# tools_v2 = [
#         {
#         # tool 2: calculate the total bill for an order
#         "type": "function",
#         "function": {
#             "name": "calculate_bill",
#             "description": "Calculate the total bill for an order.",
#             "strict": True,
#             "parameters": CalculateBillParams.model_json_schema()
#         }
#     },
#     {
#         # tool 3: submit the order request
#         "type": "function",
#         "function": {
#             "name": "payment_order",
#             "description": "Submit the order request to the restaurant's system.",
#             "strict": True,
#             "parameters": OrderSubmitParams.model_json_schema()
#         }
#     },
# ]
