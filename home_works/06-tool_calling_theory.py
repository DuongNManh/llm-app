# place_order_prompt = input("Enter your order request (e.g., 'I want to order 2 bowls of My Quang and 1 Banh Mi'): ")

# messages = [
#         {"role": "system",
#         "content": "You are a helpful product assitant for a restaurant. Always use the available tools to look up current product information and calculate bills. Do not rely on general knowledge or make up information or calculate prices manually. If the product is not found, respond with an appropriate message. If the product is found, calculate the total cost based on the restaurant's unique functions to discount. Answer in friendly conversational style."},
#         {"role": "user",
#         "content": f"{place_order_prompt}"},
# ]

# response = client.chat.completions.create(
#     model="nvidia/nemotron-3-ultra-550b-a55b:free",
#     messages=messages,
#     tools=cast(list[ChatCompletionToolUnionParam], tools),
#     tool_choice="auto",
#     temperature=0.2,
#     max_tokens=1500,
# )

# print(cast(list[ChatCompletionToolUnionParam], tools))

# print("Response from OpenRouter API:")
# print(response)
# print("================================")
# print(response.choices[0].message.content)

# assitance_message = response.choices[0].message.tool_calls
# print(assitance_message)

# print(response.choices[0].message)

# if response.choices[0].message.tool_calls:
#     for i in range(0, len(response.choices[0].message.tool_calls)):
#         print(f"-------tool call {i + 1}-------\n")
#         tool_call = response.choices[0].message.tool_calls[i]
#         args = json.loads(tool_call.function.arguments)
#         print(f"function: {tool_call.function.name}\n")
#         print(f"arguments: {args}\n")

#         result = FUNCTIONS[tool_call.function.name](**args)
#         print(f"Tool call result for {tool_call.function.name}: {result}")

#         messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)})

#     final_response = client.chat.completions.create(
#         model="nvidia/nemotron-3-ultra-550b-a55b:free",
#         messages=messages,
#         tools=cast(list[ChatCompletionToolUnionParam], tools),
#         tool_choice="auto",
#         temperature=0.2,
#         max_tokens=1500,
#     )

#     print("Final response from OpenRouter API after tool call:")
#     print(final_response)
#     print("================================")
#     print(final_response.choices[0].message.content)

# else:
#     print("No tool calls were made in the response.")
#     print(response.choices[0].message.content)
