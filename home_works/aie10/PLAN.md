# Kế hoạch cải thiện RAG - AI Engineer Roadmap Topic 10

## Mục lục
1. [Tổng quan kiến ​​trúc](#1-tổng-quan-kiến-trúc)
2. [Test suite mở rộng (Yêu cầu 1)](#2-test-suite-mở-rộng-yêu-cầu-1)
3. [Đo Baseline (Yêu cầu 2)](#3-đo-baseline-yêu-cầu-2)
4. [Kỹ thuật 1: Chunking khéo hơn (Yêu cầu 3)](#4-kỹ-thuật-1-chunking-khéo-hơn-yêu-cầu-3)
5. [Kỹ thuật 2: Hybrid Search (Yêu cầu 4)](#5-kỹ-thuật-2-hybrid-search-yêu-cầu-4)
6. [Kỹ thuật 4: Reranking (Yêu cầu 4 - alternative)](#6-kỹ-thuật-3-reranking-yêu-cầu-4)
7. [Đo từng thay đổi (Yêu cầu 5)](#7-đo-từng-thay-đổi-yêu-cầu-5)
8. [Bảng so sánh (Yêu cầu 6)](#8-bảng-so-sánh-yêu-cầu-6)
9. [Kết luận (Yêu cầu 7)](#9-kết-luận-yêu-cầu-7)

---

## 1. Tổng quan kiến trúc

### Kiến trúc hiện tại (Baseline - Topic 9)

```
PDF (huongdansudungSamSung.pdf)
  │
  ▼
PyPDFLoader ──> List[Document] (mỗi page = 1 Document)
  │
  ▼
RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
  │
  ▼
GoogleGenerativeAIEmbeddings (gemini-embedding-001)
  │
  ▼
Chroma Vector Store ──> Vector Search (k=4)
                          │
                          ▼
                    Format docs ──> LLM (gemini-1.5-flash-lite)
                          │
                          ▼
                    Answer
```

**Nhược điểm của baseline:**
1. **Chunking đơn giản** - tách theo ký tự, không tôn trọng cấu trúc PDF (section, heading, table)
2. **Pure vector search** - chỉ dùng embedding, bỏ qua keyword matching
3. **Không rerank** - top-k trực tiếp từ vector similarity, không có bước sàng lọc thứ hai
4. **k cố định (k=4)** - có thể thiếu hoặc thừa context
5. **Không in chunk lấy được** - khó debug

### Kiến trúc mục tiêu (Sau cải thiện)

```
PDF (huongdansudungSamSung.pdf)
  │
  ▼
PyPDFLoader ──> List[Document] (mỗi page = 1 Document)
  │
  ▼
[Kỹ thuật 1] Structure-Aware Chunking
  │  • Chunk_size=512, overlap=128 (giảm để precision tốt hơn)
  │  • Separators ưu tiên cấu trúc: headers -> paragraphs -> sentences
  │  • Giữ metadata: page number, source
  │
  ▼
GoogleGenerativeAIEmbeddings (gemini-embedding-001)
  │
  ├──> Chroma Vector Store (dense retrieval)
  │
  ├──> [Kỹ thuật 2a] BM25 Index (sparse retrieval) ──> Hybrid Search (weighted fusion)
  │                                                     • Vector weight: 0.5
  │                                                     • BM25 weight: 0.5
  │                                                     • top_k = 15 (lấy rộng)
  │
  ▼
[Kỹ thuật 2b] Reranker (Cross-encoder)
  │  • Input: 15 candidate chunks
  │  • Output: top 5 chunks re-ranked
  │  • Model: sentence-transformers/ms-marco-MiniLM-L-6-v2
  │
  ▼
Format docs (kèm page number, source)
  │
  ▼
LLM (gemini-1.5-flash-lite) ──> Answer
```

---

## 2. Test suite mở rộng

### Mục đích
Bộ test phải bao phủ các tình huống mà RAG dễ fail:
- **Semantic gap**: câu hỏi diễn đạt khác với từ ngữ trong tài liệu
- **Mã/tên riêng**: model numbers, feature codes
- **Out-of-scope**: câu hỏi không có trong tài liệu
- **Ambiguous**: câu hỏi mơ hồ, cần nhiều context

### Cấu trúc test case
```python
{
    "id": 1,
    "question": "...",
    "expected_answer_contains": ["..."],  # keywords bắt buộc có
    "expected_source_pages": [...],       # page numbers mong đợi
    "category": "semantic_gap" | "code_model" | "out_of_scope" | "general"
}
```

### 15 test cases

| ID | Question | Category | Expected Source Pages |
|----|----------|----------|----------------------|
| 1 | "Hãy hướng dẫn kết nối Wifi cho điện thoại Samsung" | general | ~5 |
| 2 | "Làm sao để vào internet qua Wifi trên máy Samsung?" | semantic_gap | ~5 |
| 3 | "SM-G998B có hỗ trợ sạc không dây không?" | code_model | ~3 |
| 4 | "Điện thoại Samsung bị treo logo, làm thế nào?" | general | ~10 |
| 5 | "Galaxy của tôi không khởi động được, phải làm gì?" | semantic_gap | ~10 |
| 6 | "Máy Galaxy S22 của tôi bị đơ, có cách nào cứng không?" | semantic_gap | ~10 |
| 7 | "Số IMEI là gì và xem ở đâu?" | code_model | ~2 |
| 8 | "Pin điện thoại Samsung nên sạc như thế nào cho đúng?" | general | ~7 |
| 9 | "Làm sao chụp màn hình (screenshot) trên Samsung?" | semantic_gap | ~8 |
| 10 | "Cách chuyển dữ liệu từ máy cũ sang máy Samsung mới?" | general | ~12 |
| 11 | "Tôi muốn xuất danh bạ từ Samsung ra thẻ SIM" | general | ~15 |
| 12 | "One UI 4.0 có gì mới?" | code_model | ~18 |
| 13 | "Giá bán sản phẩm là bao nhiêu?" | out_of_scope | N/A |
| 14 | "Samsung Dex là gì và cách sử dụng?" | code_model | ~20 |
| 15 | "Bảo hành điện thoại Samsung bao lâu?" | out_of_scope | N/A |

---

## 3. Đo Baseline

### Phương pháp đo
Sử dụng evaluation framework tự viết:
- **Exact match** (EM) so với expected keywords
- **Context recall**: đoạn mong đợi có xuất hiện trong retrieved chunks không
- **Accuracy**: LLM answer có chứa nội dung mong đợi không
- **Thời gian response**

### Code đo baseline
```python
def evaluate_baseline(test_cases, rag_chain):
    results = []
    for tc in test_cases:
        question = tc["question"]
        expected_pages = tc.get("expected_source_pages", [])
        
        # Lấy retrieved chunks
        retrieved_docs = retriever.get_relevant_documents(question)
        
        # Check context recall
        context_recall = check_pages_in_retrieved(retrieved_docs, expected_pages)
        
        # Get answer
        answer = rag_chain.invoke(question)
        
        # Check answer quality
        contains_keywords = all(kw in answer for kw in tc["expected_answer_contains"])
        
        results.append({
            "id": tc["id"],
            "question": question,
            "answer": answer,
            "context_recall": context_recall,
            "contains_keywords": contains_keywords,
            "retrieved_pages": [doc.metadata.get("page") for doc in retrieved_docs]
        })
    return results
```

---

## 4. Kỹ thuật 1: Chunking khéo hơn

### Lý thuyết

Chunking quyết định chất lượng retrieval. Các vấn đề với chunking hiện tại:
- **Chunk_size=1000 quá lớn**: 1 chunk chứa nhiều ý, embedding bị "dilute" (loãng)
- **Overlap=200**: có thể vẫn bỏ sót ý ở ranh giới
- **Không structure-aware**: không biết heading, section

**Các hướng cải thiện:**

1. **Reduce chunk size**: 512 thay vì 1000 -> mỗi chunk tập trung vào 1 ý
2. **Reduce overlap**: 128 -> đủ để không mất context biên
3. **Structure-aware separators**: ưu tiên `\n\n` (paragraph) > `\n` (line) > `.` (sentence)
4. **Metadata enrichment**: thêm section heading vào chunk content hoặc metadata

### Công thức tính số chunk tối ưu
```
chunk_size = average_sentence_length * average_sentences_per_paragraph * 1.5
chunk_overlap = chunk_size * 0.2  # 20% overlap
```

Với tài liệu hướng dẫn Samsung:
- Câu trung bình: ~30-50 từ
- 1 paragraph: ~5-10 câu
- => chunk_size ~ 500-750

### Implement

```python
def split_documents_improved(docs: list) -> list:
    """Structure-aware chunking với metadata enrichment."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,       # Giảm từ 1000 xuống 512
        chunk_overlap=128,    # Giảm từ 200 xuống 128
        add_start_index=True,
        separators=[
            "\n## ", "\n### ", "\n#### ",  # Headers trước
            "\n\n",                         # Paragraphs
            "\n",                           # Lines
            ".", "?", "!",                  # Sentences
            ";", ",", " ",                  # Words
            "",                             # Characters
        ],
    )
    chunks = splitter.split_documents(docs)
    
    # Enrich metadata: thêm parent page info
    for chunk in chunks:
        chunk.metadata["chunk_size"] = len(chunk.page_content)
        chunk.metadata["source_page"] = chunk.metadata.get("page", "?")
    
    return chunks
```

### Dự đoán tác động
- **Tăng precision**: chunk nhỏ hơn, tập trung hơn
- **Giảm recall**: có thể mất context nếu chunk quá nhỏ -> cần hybrid search bù vào
- **Metadata rõ ràng**: dễ debug và trích dẫn

---

## 5. Kỹ thuật 2: Hybrid Search (BM25 + Vector)

### Lý thuyết

**Vector search (dense retrieval)**:
- Embed câu hỏi và chunks vào vector space
- Tìm nearest neighbors bằng cosine similarity
- **Ưu**: hiểu ngữ nghĩa, synonym
- **Nhược**: miss khi từ khóa cụ thể (model number, code)

**BM25 (sparse retrieval)**:
- Dựa trên TF-IDF cải tiến
- Tính relevance dựa trên term frequency và document frequency
- **Ưu**: bắt chính xác từ khóa
- **Nhược**: không hiểu ngữ nghĩa

**Hybrid Search**: kết hợp cả 2
```
score = alpha * vector_score + (1-alpha) * bm25_score
```
- alpha = 0.5: cân bằng
- alpha = 0.7: thiên về semantic
- alpha = 0.3: thiên về keyword

Với dữ liệu có nhiều mã/model number -> nên tăng weight BM25.

### Implement

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

def build_hybrid_retriever(vectorstore, documents, k=15):
    """Hybrid retriever: BM25 + Vector với weighted fusion."""
    
    # BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(
        documents,
        preprocess_func=lambda x: x  # Tiếng Việt, không stem
    )
    bm25_retriever.k = k
    
    # Vector retriever
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )
    
    # Ensemble
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],  # BM25 nặng hơn do có model number
    )
    
    return hybrid_retriever
```

### Reciprocal Rank Fusion (RRF) - alternative

Nếu không muốn dùng weighted average, dùng RRF:
```
RRF_score(d) = 1/(k + rank_vector(d)) + 1/(k + rank_bm25(d))
```
với k là hằng số (thường = 60).

RRF không cần tuning weights, hoạt động tốt trong practice.

### Cấu trúc pipeline hybrid

```python
def build_hybrid_rag_chain(vectorstore, documents, k=15):
    """RAG chain với hybrid retriever."""
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-lite",
        temperature=0,
    )
    
    hybrid_retriever = build_hybrid_retriever(vectorstore, documents, k=k)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "NGỮ CẢNH:\n{context}\n\nCâu hỏi: {question}\n\nTrả lời:"),
    ])
    
    chain = (
        {"context": hybrid_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
```

---

## 6. Kỹ thuật 3: Reranking

### Lý thuyết

Reranking là bước post-processing sau retrieval:
1. **Stage 1**: retrieval nhanh (hybrid) lấy top 10-20 candidates
2. **Stage 2**: rerank chậm hơn nhưng chính xác hơn, dùng cross-encoder

**Cross-encoder vs Bi-encoder**:
- **Bi-encoder** (dùng cho retrieval): encode câu hỏi và document riêng -> cosine similarity (nhanh)
- **Cross-encoder** (dùng cho rerank): đưa cặp (question, document) qua model cùng lúc -> relevance score (chậm nhưng chính xác)

### Implement

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def build_reranker():
    """Cross-encoder reranker."""
    model = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-v2-m3"  # hoặc ms-marco-MiniLM-L-6-v2
    )
    return CrossEncoderReranker(
        model=model,
        top_n=5,  # Giữ lại top 5 chunks
    )

def build_rerank_chain(vectorstore, documents):
    """Full pipeline: hybrid retrieval + rerank."""
    
    # Stage 1: Hybrid retriever lấy 15 candidates
    hybrid_retriever = build_hybrid_retriever(vectorstore, documents, k=15)
    
    # Stage 2: Reranker
    reranker = build_reranker()
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid_retriever,
    )
    
    # Build chain
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-lite",
        temperature=0,
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "NGỮ CẢNH:\n{context}\n\nCâu hỏi: {question}\n\nTrả lời:"),
    ])
    
    chain = (
        {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
```

### Các cross-encoder models phổ biến

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `ms-marco-MiniLM-L-6-v2` | 80MB | Tốt | Nhanh |
| `BAAI/bge-reranker-v2-m3` | 2.2GB | Xuất sắc | Trung bình |
| `cross-encoder/stsb-roberta-large` | 1.5GB | Rất tốt | Chậm |

Với tài liệu tiếng Việt, nên dùng `BAAI/bge-reranker-v2-m3` vì support đa ngôn ngữ.

### Lưu ý khi rerank
- **Không gửi 10+ chunks cho LLM**: chỉ gửi top 5 sau rerank
- **In chunks lấy được**: để debug xem đoạn đúng có lọt vào candidate không

---

## 7. Đo từng thay đổi

### Quy trình đo

```
Bước 0: Baseline
  ├── Chunk: 1000/200
  ├── Retriever: Pure Vector (k=4)
  └── Result: accuracy, context_recall, latency

Bước 1: Chunking improved
  ├── Chunk: 512/128 + structure-aware
  ├── Retriever: Pure Vector (k=4)
  └── Result: so sánh với baseline

Bước 2: Hybrid Search
  ├── Chunk: 512/128 + structure-aware
  ├── Retriever: Hybrid BM25+Vector (k=15)
  └── Result: so sánh với bước 1

Bước 3: Rerank
  ├── Chunk: 512/128 + structure-aware
  ├── Retriever: Hybrid BM25+Vector (k=15) + Rerank (top_n=5)
  └── Result: so sánh với bước 2
```

### Metrics

**Context-level metrics** (trên retrieved chunks):
- **Context Recall**: tỷ lệ câu hỏi mà chunk mong đợi xuất hiện trong retrieved chunks
- **Context Precision**: tỷ lệ chunks retrieved có liên quan
- **MRR** (Mean Reciprocal Rank): rank của chunk đúng trong kết quả

**Answer-level metrics** (trên LLM answer):
- **Accuracy**: % câu trả lời đúng
- **Keyword Coverage**: % keywords mong đợi xuất hiện trong answer
- **Hallucination**: câu trả lời có nội dung không có trong context (manual check)

**Performance metrics**:
- **Latency**: thời gian response (seconds)
- **Total tokens**: số tokens sử dụng

### Code đo

```python
class RAGEvaluator:
    def __init__(self, test_cases):
        self.test_cases = test_cases
    
    def evaluate(self, chain, retriever, name=""):
        results = []
        for tc in self.test_cases:
            # Retrieval
            retrieved = retriever.invoke(tc["question"])
            retrieved_pages = [d.metadata.get("page") for d in retrieved]
            
            # Context recall
            expected_pages = tc.get("expected_source_pages", [])
            context_recall = any(p in expected_pages for p in retrieved_pages) if expected_pages else None
            
            # Answer
            answer = chain.invoke(tc["question"])
            
            # Keyword coverage
            kw = tc.get("expected_answer_contains", [])
            kw_coverage = sum(1 for k in kw if k.lower() in answer.lower()) / len(kw) if kw else None
            
            results.append({
                "question": tc["question"],
                "context_recall": context_recall,
                "kw_coverage": kw_coverage,
                "retrieved_pages": retrieved_pages,
                "answer": answer[:200],
            })
        
        # Aggregate
        cr = [r["context_recall"] for r in results if r["context_recall"] is not None]
        kc = [r["kw_coverage"] for r in results if r["kw_coverage"] is not None]
        
        return {
            "name": name,
            "context_recall": sum(cr) / len(cr) if cr else None,
            "keyword_coverage": sum(kc) / len(kc) if kc else None,
            "total_questions": len(results),
            "details": results,
        }
```

---

## 8. Bảng so sánh

### Template bảng kết quả

| Kỹ thuật | Context Recall | Keyword Coverage | Latency (s) | Δ Recall | Δ Coverage |
|----------|:-------------:|:----------------:|:-----------:|:--------:|:----------:|
| Baseline (chunk=1000, vector k=4) | 60% | 70% | 2.5 | — | — |
| + Chunking (512/128) | 65% | 73% | 2.4 | +5% | +3% |
| + Hybrid (BM25+V, k=15) | 80% | 80% | 2.8 | +15% | +7% |
| + Rerank (top_n=5) | 85% | 87% | 3.5 | +5% | +7% |

### File output

Kết quả sẽ được ghi vào `evaluation_rag.txt` và `evaluation_rag.json`:
```json
{
  "baseline": { "context_recall": 0.6, "keyword_coverage": 0.7 },
  "improved_chunking": { ... },
  "hybrid_search": { ... },
  "rerank": { ... },
  "comparison_table": "..."
}
```

---

## 9. Kết luận

### Tiêu chí đánh giá kỹ thuật nào giúp nhiều nhất

1. **Δ Context Recall**: kỹ thuật nào cải thiện recall nhiều nhất
2. **Δ Keyword Coverage**: kỹ thuật nào giúp answer chính xác hơn
3. **Cost/Latency trade-off**: kỹ thuật nào có benefit/cost tốt nhất

### Dự đoán kết quả

| Kỹ thuật | Expected Impact | Cost | Priority |
|----------|:--------------:|:----:|:--------:|
| Chunking improved | Medium | Thấp (chỉ sửa code) | 1 |
| Hybrid Search | High | Thấp (thêm BM25 index) | 2 |
| Reranking | Very High | Cao (cross-encoder) | 3 |

### Decision framework
- Nếu **budget compute thấp**: Chunking + Hybrid là đủ
- Nếu **cần accuracy cao nhất**: thêm Reranking
- Nếu **dữ liệu nhiều model number**: tăng BM25 weight

---

## Tham khảo

- [LangChain Hybrid Retrieval](https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble/)
- [LangChain Reranking](https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Cross-Encoders for Reranking](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [RAGAS Metrics](https://docs.ragas.io/en/latest/)
