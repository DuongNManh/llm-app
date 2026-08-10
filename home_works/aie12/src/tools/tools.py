from langchain_classic.tools import tool
import requests
from dotenv import load_dotenv
import os
from tavily import TavilyClient
from langchain_classic.agents import AgentExecutor, create_react_agent
from pydantic_settings import BaseSettings
from bs4 import BeautifulSoup
from readability import Document
import trafilatura #package này dùng để trích xuất nội dung văn bản từ các trang web, giúp loại bỏ các yếu tố không cần thiết như quảng cáo, menu, và các phần tử trang web khác, chỉ giữ lại nội dung chính của bài viết.

load_dotenv()

class Settings(BaseSettings):
    TAVILY_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()

tavily = TavilyClient(api_key=settings.TAVILY_API_KEY, timeout=30)

@tool
def web_search_tool(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns  Tiltles, URLs"""
    results = tavily.search(query=query, max_results=5)
    out = []
    # print(f"Web search results:\n{results}")
    for r in results["results"]:
        print(f"Title: {r['title']}, \nURL: {r['url']} \n Snippet: {r['content'][:1000]}\n")
        out.append(f"Title: {r['title']}, \nURL: {r['url']} \n Snippet: {r['content'][:1000]}\n")
    return "\n----\n".join(out)

import re


@tool  # đặt tool decorator để biến hàm này thành một công cụ có thể được sử dụng trong các agent của LangChain.
def scrape_url(url: str) -> str:
    """Scrape the content of a web page given its URL. Returns the main text content of the page.
    Use multiple extraction stragtegies for better reliability."""
    headers = {
                "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"),
                "Accept-Language": "en-US,en;q=0.9",
                "Referrer": "https://www.google.com"
            }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Raise an error for bad responses
        html_content = response.text
        
        # stragtegy 1: trafilatura (for articles and blog posts)
        extracted_content = trafilatura.extract(html_content, include_comments=False, include_tables=False, include_formatting=False)
        
        if extracted_content and len(extracted_content) > 200:
            cleaned = re.sub(r'\s+', ' ', extracted_content).strip()
            print(f"Scraped content from {url}:\n{cleaned[:2500]}")
            return cleaned[:2500]
        
        # stragtegy 2: readability (for general web pages)
        doc = Document(html_content)
        cleaned_content = doc.summary()
        soup = BeautifulSoup(cleaned_content, 'html.parser')
        
        for tag in soup([
            'script', 'style', 'header', 'footer', 'nav', 'aside', 'form', 'iframe'
        ]):
            tag.decompose() # để loại bỏ các thẻ không cần thiết
        text = soup.get_text(separator=' ', strip=True) # dòng này sẽ lấy tất cả văn bản từ trang web, loại bỏ các thẻ HTML và khoảng trắng thừa.
        if text and len(text) > 200:
            cleaned = re.sub(r'\s+', ' ', text).strip()
            print(f"Scraped content from {url}:\n{cleaned[:2500]}")
            return cleaned[:2500]
        
        # stragtegy 3: BeautifulSoup (as a last resort)
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for tag in soup([
            'script', 'style', 'header', 'footer', 'nav', 'aside', 'form', 'iframe'
        ]):
            tag.decompose() # để loại bỏ các thẻ không cần thiết
        cleaned_text = soup.get_text(separator=' ', strip=True)
        if cleaned_text and len(cleaned_text) > 200:
            cleaned = re.sub(r'\s+', ' ', cleaned_text).strip()
            print(f"Scraped content from {url}:\n{cleaned[:2500]}")
            return cleaned[:2500]
        
        else:
            return "No content could be extracted from the page."
    except requests.exceptions.Timeout as e:
        return f"Request Timeout while scraping URL: {url} {e}"
    except requests.exceptions.HTTPError as e:
        return f"HTTP Error while scraping URL: {url} {e}"
    except Exception as e:
        return f"Could not scrape URL: {url} {e}"