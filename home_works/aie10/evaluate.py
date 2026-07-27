"""Evaluation framework for RAG baseline + improvements.

Usage:
    python evaluate.py                    # baseline
    python evaluate.py --variant chunking # improved chunking
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openrouter import ChatOpenRouter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic_settings import BaseSettings

from test_suite import TEST_CASES


# ─────────── SETTINGS ───────────

class Settings(BaseSettings):
    GEMINI_API_KEY: str | None = None
    OPENROUTER_API_KEY_2: str | None = None
    model_config = {"env_file": ".env", "extra": "ignore"}

settings = Settings()

OPENROUTER_KEY = settings.OPENROUTER_API_KEY_2
if not OPENROUTER_KEY:
    raise ValueError("OPENROUTER_API_KEY_2 chưa được cấu hình trong .env")

PDF_PATH = "documents/huongdansudungSamSung.pdf"
CHROMA_DIR = "./chroma_db"
HF_EMBED_MODEL = "intfloat/multilingual-e5-base"


class E5Embeddings(Embeddings):
    """HuggingFace E5 embeddings with query/passage prefixes."""

    def __init__(self, model_name: str = HF_EMBED_MODEL):
        self.model = SentenceTransformer(model_name)
        self.encode_kwargs = {"normalize_embeddings": True}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = [f"passage: {t.replace(chr(10), ' ')}" for t in texts]
        return self.model.encode(texts, **self.encode_kwargs).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(
            [f"query: {text.replace(chr(10), ' ')}"],
            **self.encode_kwargs,
        )[0].tolist()

SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia hỗ trợ kỹ thuật và chăm sóc khách hàng của sản phẩm Samsung Smart Phone. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng bằng cách DỰA TRÊN NGỮ CẢNH được cung cấp từ tài liệu. Hãy tuân thủ nghiêm ngặt các yêu cầu sau:
1. Chỉ trả lời dựa trên thông tin trong đoạn trích. không suy đoán hoặc thêm thông tin bên ngoài.
2. Nếu đoạn trích không đủ thông tin để trả lời, hãy nói "Tài liệu không đề cập đến vấn đề này."
3. Trả lời theo nội dung trích dẫn, cùng với toàn bộ đoạn trích (nếu có) chứa câu trả lời.
4. Trả lời bằng tiếng Việt, thân thiện và dễ hiểu, không sử dụng từ ngữ chuyên ngành quá khó hiểu."""


# ─────────── CHUNKING VARIANTS ───────────

def _split_baseline(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", ".", "?", "!", ";", ",", " ", ""],
    )
    return splitter.split_documents(docs)


def _split_improved(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        add_start_index=True,
        separators=[
            "\n## ", "\n### ", "\n#### ",
            "\n\n",
            "\n",
            ".", "?", "!",
            ";", ",", " ",
            "",
        ],
    )
    return splitter.split_documents(docs)


# ─────────── INDEXING ───────────

def load_pdf(file_path: str = PDF_PATH) -> list:
    loader = PyPDFLoader(file_path)
    return loader.load()


def create_embeddings():
    return E5Embeddings()


def create_vectorstore(documents: list, persist_dir: str = CHROMA_DIR) -> Chroma:
    import shutil
    if Path(persist_dir).exists():
        shutil.rmtree(persist_dir)
    embeddings = create_embeddings()
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    total = len(documents)
    batch_size = 10
    for i in range(0, total, batch_size):
        batch = documents[i : i + batch_size]
        vectorstore.add_documents(batch)
        print(f"  → Batch {i//batch_size + 1}/{(total-1)//batch_size + 1}")
    return vectorstore


def load_vectorstore(persist_dir: str = CHROMA_DIR) -> Chroma:
    embeddings = create_embeddings()
    return Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


# ─────────── RETRIEVAL VARIANTS ───────────

def build_vector_retriever(vectorstore, k: int = 4):
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_hybrid_retriever(vectorstore, documents, k: int = 15):
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )


# ─────────── FORMAT & LLM ───────────

def format_docs(docs: list) -> str:
    return "\n\n---\n\n".join(
        f"[Trang {doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in docs
    )


def build_llm():
    return ChatOpenRouter(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        temperature=0,
        api_key=OPENROUTER_KEY,
    )


def build_chain(retriever):
    llm = build_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "NGỮ CẢNH:\n{context}\n\nCâu hỏi: {question}\n\nTrả lời:"),
    ])
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


# ─────────── EVALUATION ───────────

def normalize_text(text: str) -> str:
    return text.lower().strip()


def keyword_coverage(answer: str, expected_keywords: list) -> float:
    if not expected_keywords:
        return 1.0
    answer_lower = normalize_text(answer)
    matches = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return matches / len(expected_keywords)


def check_context_recall(retrieved_docs: list, expected_pages: list) -> bool:
    if not expected_pages:
        return True
    retrieved_pages = set()
    for doc in retrieved_docs:
        p = doc.metadata.get("page")
        if p is not None:
            try:
                retrieved_pages.add(int(p) + 1)
            except (ValueError, TypeError):
                retrieved_pages.add(p)
    return any(ep in retrieved_pages for ep in expected_pages)


def run_evaluation(
    variant_name: str,
    retriever,
    chain,
    test_cases: list,
    use_llm: bool = True,
) -> dict:
    results = []
    total_cr = 0
    total_kw = 0.0
    total_latency = 0.0
    cr_count = 0
    kw_count = 0

    for tc in test_cases:
        question = tc["question"]
        expected_kw = tc.get("expected_answer_contains", [])
        expected_pages = tc.get("expected_source_pages", [])

        t0 = time.time()
        retrieved_docs = retriever.invoke(question)
        t1 = time.time()

        if use_llm:
            answer = chain.invoke(question)
            t2 = time.time()
        else:
            answer = "(skipped - no LLM)"
            t2 = t1

        latency_total = t2 - t0

        cr = check_context_recall(retrieved_docs, expected_pages)
        kw = keyword_coverage(answer, expected_kw) if use_llm else (0.0 if expected_kw else 1.0)

        retrieved_page_list = []
        for d in retrieved_docs:
            p = d.metadata.get("page")
            if p is not None:
                try:
                    retrieved_page_list.append(int(p) + 1)
                except (ValueError, TypeError):
                    retrieved_page_list.append(p)
            else:
                retrieved_page_list.append(None)

        results.append({
            "id": tc["id"],
            "category": tc["category"],
            "question": question,
            "answer_preview": answer[:400],
            "context_recall": cr,
            "keyword_coverage": round(kw, 4),
            "latency_total": round(latency_total, 3),
            "retrieved_pages": retrieved_page_list,
            "expected_pages": expected_pages,
        })

        if expected_pages:
            total_cr += int(cr)
            cr_count += 1
        if use_llm and expected_kw:
            total_kw += kw
            kw_count += 1
        total_latency += latency_total

        kw_str = f"{kw:.2f}" if use_llm else "N/A"
        status = "✓" if (cr if expected_pages else True) else "✗"
        print(f"  [{status}] #{tc['id']} {tc['category']:14s} | CR={cr} KW={kw_str} "
              f"| pages={retrieved_page_list} | {latency_total:.1f}s")

    n = len(results)
    kw_avg = round(total_kw / kw_count, 4) if (use_llm and kw_count) else None
    return {
        "variant": variant_name,
        "num_cases": n,
        "context_recall": round(total_cr / cr_count, 4) if cr_count else None,
        "keyword_coverage": kw_avg,
        "avg_latency": round(total_latency / n, 3) if n else None,
        "details": results,
    }


# ─────────── PIPELINE SETUP ───────────

def build_variant(variant: str, reuse_db: bool = False):
    if reuse_db and Path(CHROMA_DIR).exists():
        docs = load_pdf()
        if variant == "baseline":
            chunks = _split_baseline(docs)
        else:
            chunks = _split_improved(docs)
        vectorstore = load_vectorstore()
    else:
        docs = load_pdf()
        if variant == "baseline":
            chunks = _split_baseline(docs)
        else:
            chunks = _split_improved(docs)
        print(f"  Chunks: {len(chunks)}")
        vectorstore = create_vectorstore(chunks)

    if variant in ("hybrid", "rerank"):
        retriever = build_hybrid_retriever(vectorstore, chunks, k=15)
    else:
        retriever = build_vector_retriever(vectorstore, k=4)

    chain = build_chain(retriever)
    return retriever, chain


def save_report(result: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    txt_path = path.replace(".json", ".txt")
    lines = [
        f"{'='*60}",
        f"EVALUATION REPORT: {result['variant']}",
        f"{'='*60}",
        f"  Total cases : {result['num_cases']}",
        f"  Context Rec : {result['context_recall']}",
        f"  Keyword Cov : {result['keyword_coverage']}",
        f"  Avg Latency : {result['avg_latency']}s",
        f"{'='*60}",
        "",
    ]
    for r in result["details"]:
        status = "✓" if (r["context_recall"] if r["expected_pages"] else True) else "✗"
        lines.append(
            f"#{r['id']} [{status}] {r['category']:14s}"
            f"  CR={r['context_recall']}  KW={r['keyword_coverage']:.2f}"
            f"  {r['latency_total']:.1f}s"
        )
        lines.append(f"     pages -> {r['retrieved_pages']} (expected {r['expected_pages']})")
        lines.append(f"     Q: {r['question'][:80]}")
        lines.append(f"     A: {r['answer_preview'][:200]}")
        lines.append("")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  → Saved {path} and {txt_path}")


# ─────────── MAIN ───────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="baseline",
                        choices=["baseline", "chunking", "hybrid", "rerank"])
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM calls, evaluate retrieval only")
    parser.add_argument("--reuse-db", action="store_true",
                        help="Reuse existing chroma_db if available")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Variant: {args.variant}" + (" (retrieval only)" if args.no_llm else ""))
    print(f"{'='*60}")

    retriever, chain = build_variant(args.variant, reuse_db=args.reuse_db)

    print(f"\nEvaluating {len(TEST_CASES)} test cases...\n")
    result = run_evaluation(args.variant, retriever, chain, TEST_CASES, use_llm=not args.no_llm)

    kw_str = f"{result['keyword_coverage']}" if result['keyword_coverage'] is not None else "N/A"
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.variant}")
    print(f"  Context Recall : {result['context_recall']}")
    print(f"  Keyword Coverage: {kw_str}")
    print(f"  Avg Latency    : {result['avg_latency']}s")
    print(f"{'='*60}\n")

    suffix = "_retrieval" if args.no_llm else ""
    out_path = f"evaluation_aie10_{args.variant}{suffix}.json"
    save_report(result, out_path)


if __name__ == "__main__":
    main()
