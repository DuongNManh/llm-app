import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_classic.prompts import PromptTemplate
import streamlit as st
from pydantic_settings import BaseSettings
import requests
# from langchain.tools import tool

from langchain_classic.tools import tool

from langchain_community.tools.tavily_search import TavilySearchResults

# .env nằm ở thư mục gốc dự án (2 cấp so với file này), dùng path tuyệt đối để
# không phụ thuộc vào thư mục đang chạy (quan trọng khi chạy `streamlit run`)
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

print("GEMINI_API_KEY:", GEMINI_API_KEY)
print("TAVILY_API_KEY:", TAVILY_API_KEY)


## ==== Load environment variables from .env file ====
class Settings(BaseSettings):
    GEMINI_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_API_KEY_2: str | None = None
    TAVILY_API_KEY: str | None = None
    model_config = {"env_file": str(ENV_PATH), "extra": "ignore"}
    
settings = Settings()


## ===== setup streamlit page  =======

st.set_page_config(
    page_title="AIe11",
    layout="centered"
)

st.title("AIe11 - AI Agents Playground")
st.markdown("search + weather api agent using LangChain")


## ===== search tool và weather tool ===== 

# weather tool
@tool
def get_weather_data(city: str):
    """lấy nhiệt độ hiện tại của một thành phố"""
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": city,
            "count": 1,
            "language": "vi",
            "format": "json"
        },
        timeout=10
    ).json()

    if not geo.get("results"):
        raise ValueError(f"City '{city}' not found.")

    location = geo["results"][0]

    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current_weather": "true"
        },
        timeout=10,
    ).json()
    print(weather)
    temperature = weather.get("current_weather", {}).get("temperature")
    return {"city": city, "temperature": temperature}

## ====== setup llm, agent =======

from langchain_google_genai import ChatGoogleGenerativeAI

# Định nghĩa template chuỗi (nội dung chi tiết có tại hwchase17/react)
react_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

# Các object nặng (tool, llm, agent, executor) chỉ khởi tạo 1 lần và được
# cache lại cho mọi lần rerun nhờ @st.cache_resource
@st.cache_resource
def build_tavily_tool():
    return TavilySearchResults(tavily_api_key=TAVILY_API_KEY, max_results=2)


@st.cache_resource
def build_agent_executor():
    from langchain_classic.agents import create_react_agent, AgentExecutor

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        api_key=GEMINI_API_KEY,
        temperature=0.0,
    )
    prompt = PromptTemplate.from_template(react_template)
    search_tool = build_tavily_tool()

    agent = create_react_agent(
        llm=llm,
        tools=[search_tool, get_weather_data],
        prompt=prompt,
    )
    return AgentExecutor(
        agent=agent,
        tools=[search_tool, get_weather_data],
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        verbose=True,
    )


# Khởi tạo executor một lần (các lần rerun sau lấy lại từ cache)
agent_executor = build_agent_executor()

# ======== UI Input ===========

user_input = st.text_input("Enter your query:", placeholder="Ask me anything...")

# ========= Run Agent ==========

if st.button("Submit"):
    if user_input:
        with st.spinner("Agent is thinking..."):
            try:
                res = agent_executor.invoke({"input": user_input})
                st.success("Agent completed the task!")
                st.markdown("### Final Response:")
                st.write(res["output"])
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a query before submitting.")
