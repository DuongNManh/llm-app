"""
bikipchunking.py — Semantic Chunking cho Samsung manual

Pipeline:
    load PDF → tách câu (kèm metadata) → combine câu với buffer
    → embed combined sentences → tính cosine distance giữa các câu liên tiếp
    → tìm breakpoint (ngưỡng percentile 95) → gộp thành chunk
    → visualize + in thử chunk
"""

# ═══════════════════════════════════════════════════════════════
# 1. Imports & config
# ═══════════════════════════════════════════════════════════════

import re
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

PDF_PATH = "documents/huongdansudungSamSung.pdf"

# ═══════════════════════════════════════════════════════════════
# 2. PDF loading
# ═══════════════════════════════════════════════════════════════


def load_pdf_docs(pdf_path: str = PDF_PATH):
    loader = PyPDFLoader(pdf_path)
    return loader.load()


# ═══════════════════════════════════════════════════════════════
# 3. Sentence extraction (kèm metadata đề mục + index)
# ═══════════════════════════════════════════════════════════════


def _detect_running_headers(docs) -> set[str]:
    """Detect running header theo thống kê: dòng ngắn xuất hiện >= 3 lần ở đầu trang."""
    candidates = Counter() # sử dụng Counter để đếm số lần xuất hiện của các dòng đầu trang
    # duyệt qua tất cả các trang
    for doc in docs:
        # loại bỏ các dòng trống và các dòng chỉ chứa số trang
        # l.strip() để loại bỏ khoảng trắng đầu và cuối dòng
        # doc.page_content.split("\n") để tách nội dung trang thành các dòng
        # if l.strip() để chỉ giữ lại các dòng không trống
        lines = [l.strip() for l in doc.page_content.split("\n") if l.strip()]
        # skip trang trống không có nội dung
        if not lines:
            continue
        # nếu dòng đầu tiên là số trang, bỏ qua dòng đó
        # tại sao? Bởi vì số trang không phải là running header, nên chúng ta không muốn đếm nó như một đề mục.
        if lines[0].isdigit():
            lines = lines[1:]
        if not lines:
            continue
        first = lines[0]
        if (len(first) < 40
                and not first.isdigit()
                and not first.startswith(("•", "●", "-", "*"))):
            candidates[first] += 1
    return {h for h, c in candidates.items() if c >= 3}


def _strip_page_furniture(lines: list[str], page_num: int, running_headers: set[str]) -> tuple[list[str], str | None]:
    """Xóa running header + số trang ở đầu trang.

    Returns:
        (cleaned_lines, header): header là đề mục đã bị strip (None nếu không có).
    """
    printed = str(page_num + 1)  # PyPDFLoader 0-indexed, số in trên giấy = page + 1
    header = None
    if not lines:
        return lines, None

    # pattern [header][pagenum][content...]
    if len(lines) >= 2 and lines[1] == printed:
        header = lines[0]
        return lines[2:], header
    # pattern [pagenum][...]
    if lines[0] == printed:
        lines = lines[1:]
    # pattern [pagenum?][header][content...] — dòng đầu trùng running header
    if lines and lines[0] in running_headers:
        header = lines[0]
        return lines[1:], header
    return lines, None


def _is_toc_page(page_text: str) -> bool:
    """Trang TOC chứa nhiều số trang rời rạc (dấu hiệu 'mục lục')."""
    numbers = re.findall(r"(?<!\w)\d+(?!\w)", page_text)
    return len(numbers) >= 10


def _split_sentences(page_text: str) -> list[tuple[int, str]]:
    """Tách câu, trả về [(char_start, text), ...]."""
    sentences = []
    for m in re.finditer(r'[^.!?]+[.!?]+(?:\s+|$)|[^.!?]+$', page_text):
        text = m.group().strip()
        if text:
            sentences.append((m.start(), text))
    return sentences


def extract_sentences_from_docs(docs):
    """Tách câu toàn tài liệu với index toàn cục + index trong trang + metadata đề mục.

    Mỗi câu trả về dict:
        {
            "sentence_id": int,          # index toàn cục, câu thứ bao nhiêu trong tài liệu (1-based)
            "page": int,                 # số trang (0-indexed từ PyPDFLoader)
            "page_sentence_index": int,  # câu thứ bao nhiêu trong trang (1-based)
            "char_start": int,           # vị trí bắt đầu câu trong page_text đã clean
            "section": str,              # đề mục (running header) của trang đang chứa câu
            "text": str,
        }
    """
    all_sentences = []
    global_idx = 0
    current_section = ""
    running_headers = _detect_running_headers(docs)

    for doc in docs:
        page_num = doc.metadata.get("page", 0)
        lines = [l.strip() for l in doc.page_content.split("\n") if l.strip()]

        cleaned_lines, header = _strip_page_furniture(lines, page_num, running_headers)
        if header:
            current_section = header

        page_text = " ".join(cleaned_lines)

        # bỏ qua trang bìa (page 0) và các trang TOC
        if page_num == 0 or _is_toc_page(doc.page_content):
            continue

        page_sentences = _split_sentences(page_text)

        for local_idx, (char_start, sentence) in enumerate(page_sentences, start=1):
            global_idx += 1
            all_sentences.append({
                "sentence_id": global_idx,
                "page": page_num,
                "page_sentence_index": local_idx,
                "char_start": char_start,
                "section": current_section,
                "text": sentence,
            })

    return all_sentences


# ═══════════════════════════════════════════════════════════════
# 4. Semantic chunking
# ═══════════════════════════════════════════════════════════════


def combine_sentences(sentences: list[dict], buffer_size: int = 1) -> list[dict]:
    """Gom buffer_size câu trước + câu hiện tại + buffer_size câu sau thành 1 chuỗi ngữ cảnh.

    Gán thêm key `combined_sentence` vào từng dict. Chuỗi này được dùng để embed
    vì câu đơn lẻ thường quá ngắn, thiếu ngữ cảnh khi tính độ tương đồng.
    """
    for i in range(len(sentences)):
        context_parts = []
        for j in range(i - buffer_size, i + buffer_size + 1):
            if 0 <= j < len(sentences):
                context_parts.append(sentences[j]["text"])
        sentences[i]["combined_sentence"] = " ".join(context_parts)
    return sentences


def embed_combined_sentences(sentences: list[dict], embeddings_model) -> list[dict]:
    """Embed toàn bộ combined_sentence, lưu vào key `combined_sentence_embedding`."""
    combined = [s["combined_sentence"] for s in sentences]
    embeddings = embeddings_model.embed_documents(combined)
    for s, emb in zip(sentences, embeddings):
        s["combined_sentence_embedding"] = emb
    return sentences


def calculate_cosine_distances(sentences: list[dict]) -> list[float]:
    """Tính cosine distance giữa câu i và câu i+1 (dùng embedding đã có).

    distance = 1 - cosine_similarity.
    Gán `distance_to_next` cho từng dict và trả về danh sách distance.
    Distance lớn = ranh giới chủ đề → nơi nên tách chunk.
    """
    distances = []
    for i in range(len(sentences) - 1):
        emb_current = sentences[i]["combined_sentence_embedding"]
        emb_next = sentences[i + 1]["combined_sentence_embedding"]
        similarity = cosine_similarity([emb_current], [emb_next])[0][0]
        distance = 1 - similarity
        distances.append(distance)
        sentences[i]["distance_to_next"] = distance
    return distances


def find_chunk_breakpoints(distances: list[float], percentile: int = 95) -> tuple[float, list[int]]:
    """Tìm vị trí tách chunk dựa trên ngưỡng percentile.

    Returns:
        (threshold, indices_above_thresh)
        - threshold: giá trị distance tại percentile (dòng đỏ trên biểu đồ)
        - indices_above_thresh: index các distance vượt ngưỡng → ranh giới chunk
    """
    # tính ngưỡng percentile
    # biến distances là danh sách các khoảng cách cosine giữa các câu liên tiếp
    threshold = float(np.percentile(distances, percentile))
    # list các index sentence mà distance vượt ngưỡng → ranh giới chunk
    indices_above_thresh = [i for i, d in enumerate(distances) if d > threshold]
    return threshold, indices_above_thresh


def build_chunks(sentences: list[dict], breakpoints: list[int]) -> list[dict]:
    """Gộp các câu thành chunk theo breakpoints, kèm metadata traceback.

    Mỗi chunk trả về dict:
        {
            "chunk_id": int,
            "sentence_ids": [int, ...],   # các sentence_id gộp vào chunk
            "section": str,               # section của chunk
            "page_start": int, "page_end": int,
            "text": str,
        }
    """
    chunks = []
    start_idx = 0
    boundaries = breakpoints + [len(sentences) - 1]

    for chunk_id, end_idx in enumerate(boundaries):
        group = sentences[start_idx:end_idx + 1]
        if not group:
            continue
        chunks.append({
            "chunk_id": chunk_id,
            "sentence_ids": [s["sentence_id"] for s in group],
            "section": group[0]["section"],
            "page_start": group[0]["page"],
            "page_end": group[-1]["page"],
            "text": " ".join(s["text"] for s in group),
        })
        start_idx = end_idx + 1

    return chunks


def visualize_breakpoints(distances: list[float], breakpoints: list[int], threshold: float,
                          title: str = "Semantic Chunks", y_upper_bound: float = 0.2):
    """Vẽ biểu đồ cosine distance giữa các câu liên tiếp + tô màu từng chunk.

    - Đường đỏ: ngưỡng percentile (chỗ nào vượt là ranh giới chunk)
    - Các vùng tô màu: từng chunk
    """
    plt.figure(figsize=(14, 5))
    plt.plot(distances)

    plt.ylim(0, y_upper_bound)
    plt.xlim(0, len(distances))
    plt.axhline(y=threshold, color="r", linestyle="-")

    num_chunks = len(breakpoints) + 1
    plt.text(x=(len(distances) * 0.01), y=y_upper_bound / 50,
             s=f"{num_chunks} Chunks")

    colors = ["b", "g", "r", "c", "m", "y", "k"]
    for i, bp in enumerate(breakpoints):
        start = 0 if i == 0 else breakpoints[i - 1]
        end = bp if i < len(breakpoints) - 1 else len(distances)
        plt.axvspan(start, end, facecolor=colors[i % len(colors)], alpha=0.25)
        plt.text(x=np.average([start, end]),
                 y=threshold + y_upper_bound / 20,
                 s=f"Chunk #{i}", horizontalalignment="center", rotation="vertical")

    # tô từ breakpoint cuối tới hết dataset
    if breakpoints:
        last = breakpoints[-1]
        plt.axvspan(last, len(distances), facecolor=colors[(len(breakpoints)) % len(colors)], alpha=0.25)
        plt.text(x=np.average([last, len(distances)]),
                 y=threshold + y_upper_bound / 20,
                 s=f"Chunk #{len(breakpoints)}", rotation="vertical")

    plt.title(title)
    plt.xlabel("Index of sentences (sentence position)")
    plt.ylabel("Cosine distance between sequential sentences")
    plt.show()


# ═══════════════════════════════════════════════════════════════
# 5. Pipeline orchestration
# ═══════════════════════════════════════════════════════════════


def run_semantic_chunking(
    sentences: list[dict],
    buffer_size: int = 1,
    percentile: int = 95,
    visualize: bool = True,
) -> tuple[list[dict], list[float], list[int]]:
    """Chạy toàn bộ pipeline semantic chunking.

    Returns:
        (chunks, distances, breakpoints)
    """
    print(f"[1/4] Combining sentences (buffer={buffer_size})...")
    sentences = combine_sentences(sentences, buffer_size)

    print("[2/4] Embedding combined sentences...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cpu"},
        show_progress=True,
        encode_kwargs={"normalize_embeddings": True},
    )
    sentences = embed_combined_sentences(sentences, embeddings_model)

    print("[3/4] Calculating cosine distances...")
    distances = calculate_cosine_distances(sentences)

    print(f"[4/4] Finding breakpoints (percentile={percentile})...")
    threshold, breakpoints = find_chunk_breakpoints(distances, percentile)
    chunks = build_chunks(sentences, breakpoints)
    print(f"  => {len(chunks)} chunks, {len(sentences)} sentences")

    if visualize:
        visualize_breakpoints(distances, breakpoints, threshold)

    return chunks, distances, breakpoints


# ═══════════════════════════════════════════════════════════════
# 6. Main entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading PDF...")
    docs = load_pdf_docs()
    sentences = extract_sentences_from_docs(docs)
    print(f"  => {len(sentences)} sentences")

    chunks, distances, breakpoints = run_semantic_chunking(sentences, buffer_size=2, percentile=80, visualize=True)

    print("\nPreview 2 chunks:")
    for chunk in chunks:
        print(f"\nChunk #{chunk['chunk_id']} | section={chunk['section']} "
              f"| sentences {chunk['sentence_ids'][0]}..{chunk['sentence_ids'][-1]} "
              f"| pages {chunk['page_start']}..{chunk['page_end']}")
        print(chunk["text"])
        print(len(chunk["text"]), "characters")
        print("-" * 80)
