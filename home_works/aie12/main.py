import streamlit as st

from src.pipelines.pipeline import (
    research_pipeline,
)


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# Suggested topics
# ============================================================

SUGGESTED_TOPICS = [
    "LangGraph and AI Agent architecture",
    "Semantic Chunking for RAG",
    "Hybrid Search with BM25 and Vector Search",
    "RAG evaluation techniques",
    "AI Agent memory architecture",
]


# ============================================================
# Header
# ============================================================

st.title("🔎 AI Research Assistant")

st.markdown(
    """
Research a topic using:

**Web Search → Web Scraping → Report Generation → Critic Review**
"""
)


# ============================================================
# Input
# ============================================================

topic = st.text_input(
    "Research topic",
    placeholder="Enter a topic you want to research...",
)


st.caption("Suggested topics")

cols = st.columns(len(SUGGESTED_TOPICS))

for col, suggested_topic in zip(cols, SUGGESTED_TOPICS):

    if col.button(
        suggested_topic,
        use_container_width=True,
    ):
        topic = suggested_topic
        st.session_state["topic"] = suggested_topic


# Restore selected topic
if "topic" in st.session_state and not topic:
    topic = st.session_state["topic"]


# ============================================================
# Start button
# ============================================================

start = st.button(
    "🚀 Start Research",
    type="primary",
    use_container_width=True,
)


# ============================================================
# Research execution
# ============================================================

if start:

    if not topic.strip():
        st.warning("Please enter a research topic.")
        st.stop()

    st.divider()

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    progress_bar = st.progress(0)

    status_placeholder = st.empty()

    step_status = {
        1: st.empty(),
        2: st.empty(),
        3: st.empty(),
        4: st.empty(),
    }

    def update_progress(step: int, message: str):

        progress_bar.progress(
            step / 4
        )

        status_placeholder.info(
            f"Step {step}/4 — {message}"
        )

        for i in range(1, 5):

            if i < step:
                step_status[i].markdown(
                    f"✅ **Step {i} completed**"
                )

            elif i == step:
                step_status[i].markdown(
                    f"🔄 **Step {i} — {message}**"
                )

            else:
                step_status[i].markdown(
                    f"⏳ Step {i} — Waiting"
                )


    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    for i in range(1, 5):
        step_status[i].markdown(
            f"⏳ Step {i} — Waiting"
        )


    # --------------------------------------------------------
    # Run pipeline
    # --------------------------------------------------------

    try:

        state = research_pipeline(
            topic,
            progress_callback=update_progress,
        )

    except Exception as e:

        st.error(
            f"Research pipeline failed: {str(e)}"
        )

        st.stop()


    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    progress_bar.progress(1.0)

    status_placeholder.success(
        "Research completed successfully!"
    )


    # ========================================================
    # Results
    # ========================================================

    st.divider()

    st.subheader("📚 Research Process")


    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------

    with st.expander(
        "🔍 Step 1 — Web Search",
        expanded=False,
    ):

        st.markdown(
            "### Search Results"
        )

        st.markdown(
            state["search_results"]
        )


    # --------------------------------------------------------
    # Step 2
    # --------------------------------------------------------

    with st.expander(
        "🌐 Step 2 — Web Scraping",
        expanded=False,
    ):

        st.markdown(
            "### Scraped Content"
        )

        st.markdown(
            state["scrape_results"]
        )


    # --------------------------------------------------------
    # Step 3
    # --------------------------------------------------------

    with st.expander(
        "📝 Step 3 — Research Report",
        expanded=True,
    ):

        st.markdown(
            state["report"]
        )


    # --------------------------------------------------------
    # Step 4
    # --------------------------------------------------------

    with st.expander(
        "🔎 Step 4 — Critic Review",
        expanded=True,
    ):

        st.markdown(
            state["feedback"]
        )