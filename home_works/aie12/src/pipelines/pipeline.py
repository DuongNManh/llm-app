from src.agents.agent import (
    build_search_agent,
    build_reader_agent,
    write_chain,
    critic_chain,
)


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        text_parts.append(text)

        return "\n".join(text_parts)

    return str(content)


def research_pipeline(
    topic: str,
    progress_callback=None,
) -> dict:

    state = {}

    def progress(step: int, message: str):
        if progress_callback:
            progress_callback(step, message)

    # ============================================================
    # Step 1: Web Search
    # ============================================================

    progress(1, "Searching the web...")

    search_agent = build_search_agent()

    search_results = search_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Find recent, reliable and detailed information "
                        f"about the topic: {topic}"
                    ),
                }
            ]
        }
    )

    last_search_message = search_results["messages"][-1]

    state["search_results"] = _content_to_text(
        last_search_message.content
    )

    progress(1, "Web search completed.")

    # ============================================================
    # Step 2: Web Scraping
    # ============================================================

    progress(2, "Selecting and scraping the most relevant source...")

    reader_agent = build_reader_agent()

    reader_results = reader_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Based on the following search results about {topic}, "
                        f"pick the most relevant URL and scrape it for deeper "
                        f"content.\n\n"
                        f"Search Results:\n"
                        f"{state['search_results'][:1500]}"
                    ),
                }
            ]
        }
    )

    last_reader_message = reader_results["messages"][-1]

    state["scrape_results"] = _content_to_text(
        last_reader_message.content
    )

    progress(2, "Web scraping completed.")

    # ============================================================
    # Step 3: Write Research Report
    # ============================================================

    progress(3, "Writing research report...")

    research_combined = (
        f"SEARCH RESULTS:\n"
        f"{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n"
        f"{state['scrape_results']}\n\n"
    )

    state["report"] = write_chain.invoke(
        {
            "topic": topic,
            "research": research_combined,
        }
    )

    progress(3, "Research report completed.")

    # ============================================================
    # Step 4: Critic
    # ============================================================

    progress(4, "Critically reviewing the research report...")

    state["feedback"] = critic_chain.invoke(
        {
            "report": state["report"],
        }
    )

    progress(4, "Critic review completed.")

    return state