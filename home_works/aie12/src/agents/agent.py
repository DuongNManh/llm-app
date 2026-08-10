from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import web_search_tool, scrape_url
from langchain.agents import create_agent
from dotenv import load_dotenv
import os
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    OPEN_ROUTER_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    TAVILY_API_KEY: str | None = None
    model_config = {"env_file": ".env", "extra": "ignore"}

settings = Settings()
MODEL = "gemini-3.1-flash-lite"

llm = ChatGoogleGenerativeAI(
    model=MODEL,
    api_key=settings.GEMINI_API_KEY,
    temperature=0.0,
)

def build_search_agent():
    """Build LangChain agent specifically for web search tasks."""
    return create_agent(
        model=llm,
        tools=[web_search_tool],
        system_prompt="You're a web search agent. Your task is to find recent, reliable and detailed information about a given topic. Use the provided web search tool to gather information, not summarizing. Return the following format: Title: {title}, URL: {url}, Snippet: {snippet}. Be concise and accurate in your responses."
    )

def build_reader_agent():
    """Build LangChain agent specifically for web scraping tasks."""
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="You're a web scraping agent. Your task is to scrape the content of a web page given its URL. Use the provided web scraping tool to extract the main text content of the page. Return the content in a clean and readable format. Be concise and accurate in your responses."
    )

# write chain - chain cho agent thực hiện viết báo cáo nghiên cứu dựa trên thông tin đã thu thập được từ web search và scrape.

WRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert researcger writer. Writer clear, structured and insightful reports."),
    ("human", """write a detailed research report on the topic below.\n
    Topic: {topic}\n
    Research Gathered:\n
    {research}\n
    Structured the report as:
    -Introduction
    -Key Findings (minimum 3 well explained points)
    -Conclusion
    -Sources (list all URLs found in the research)
    Be detailed, factual and professional. Answer in Vietnamese.""")
])

write_chain = WRITE_PROMPT | llm | StrOutputParser() # vì sao nó biết là Runable cho biến chain này? Trả lời: Vì nó là một chuỗi các bước được kết nối với nhau, và mỗi bước đều là một Runable.


# critic chain - dùng để đánh giá chất lượng của báo cáo nghiên cứu được tạo ra bởi write_chain.

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific in your feedback."),
    ("human", """Review the research report below and evaluate it strictly.
    Report:\n
    {report}\n
    
    Response in this exact format:
    Score: X/10
    Strengths: (list the strengths of the report)
    -...
    -...
    Areas for Improvement: (list the areas where the report can be improved)
    -...
    -...
    One line verdict: (provide a concise overall assessment of the report)
    """)
])

critic_chain = CRITIC_PROMPT | llm | StrOutputParser()
