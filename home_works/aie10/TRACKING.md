# TRACKING - Cải thiện RAG Topic 10

## Trạng thái tổng quan

| Step | Task | Status | Date | Notes |
|------|------|--------|------|-------|
| 0 | Tạo kế hoạch & architecture | ✅ Done | 2026-07-27 | File: PLAN.md |
| 1 | Mở rộng test suite (10-15 câu) | ⬜ Pending | | |
| 2 | Đo baseline | ⬜ Pending | | |
| 3 | Kỹ thuật 1: Chunking khéo hơn | ⬜ Pending | | |
| 4 | Đo lại sau chunking | ⬜ Pending | | |
| 5 | Kỹ thuật 2: Hybrid Search (BM25+Vector) | ⬜ Pending | | |
| 6 | Đo lại sau hybrid | ⬜ Pending | | |
| 7 | Kỹ thuật 3: Reranking | ⬜ Pending | | |
| 8 | Đo lại sau rerank | ⬜ Pending | | |
| 9 | Tổng hợp bảng so sánh | ⬜ Pending | | |
| 10 | Kết luận | ⬜ Pending | | |

---

## Chi tiết từng bước

### Bước 1: Mở rộng test suite
**File đích:** `home_works/aie10/test_suite.py`

Cần làm:
- [ ] Định nghĩa 15 test cases (5 semantic_gap, 4 code_model, 4 general, 2 out_of_scope)
- [ ] Mỗi test case có: question, expected_answer_contains, expected_source_pages, category
- [ ] Xác thực expected_source_pages bằng cách đọc PDF gốc

**Ghi chú:**

---

### Bước 2: Đo baseline
**File đích:** `home_works/aie10/evaluate.py`

Cần làm:
- [ ] Copy `index.py` thành `baseline_index.py` để giữ baseline gốc
- [ ] Viết evaluator: context_recall, keyword_coverage, latency
- [ ] Chạy baseline trên test suite
- [ ] Ghi kết quả vào `evaluation_aie10_baseline.json`

**Kết quả:**
| Metric | Value |
|--------|-------|
| Context Recall | 84.62% |
| Keyword Coverage | 92.22% |
| Avg Latency | 10.78s |

**Ghi chú:** CR fails on #6 và #7 (treo logo/đơ máy) — chunk quá lớn (1000) làm page 19 bị merge với nội dung khác, vector search không match đúng. KW vẫn 1.00 vì LLM suy luận từ troubleshooting pages khác.

---

### Bước 3: Chunking khéo hơn
**File đích:** `home_works/aie10/pipeline_v1_chunking.py`

Cần làm:
- [ ] Tham số mới: chunk_size=512, chunk_overlap=128
- [ ] Separators ưu tiên: headers -> paragraphs -> sentences
- [ ] Metadata enrichment (source_page, chunk_size)
- [ ] Giữ nguyên vector search (chưa hybrid/rerank)
- [ ] Chạy evaluation, ghi kết quả `evaluation_aie10_v1_chunking.json`

**Kết quả:**
| Metric | Value | Δ vs Baseline |
|--------|-------|:-------------:|
| Context Recall | | |
| Keyword Coverage | | |
| Avg Latency | | |

**Ghi chú:**

---

### Bước 4: Hybrid Search
**File đích:** `home_works/aie10/pipeline_v2_hybrid.py`

Cần làm:
- [ ] Thêm BM25Retriever từ langchain_community
- [ ] EnsembleRetriever với weights [0.5, 0.5]
- [ ] Tăng top_k lên 15 (lấy rộng cho rerank sau)
- [ ] Giữ nguyên chunking từ v1
- [ ] Chạy evaluation, ghi kết quả `evaluation_aie10_v2_hybrid.json`

**Kết quả:**
| Metric | Value | Δ vs Chunking |
|--------|-------|:--------------:|
| Context Recall | 92.31% | +7.69% |
| Keyword Coverage | N/A (no LLM) | - |
| Avg Latency | 0.068s | -10.99s |

**Ghi chú:** Hybrid (BM25+Vector, k=15) cải thiện CR đáng kể — #6 (treo logo) đã được fix nhờ BM25 tìm thấy page 19. #7 (bị đơ) vẫn fail vì query không có keyword chung với page 19. Avg latency giảm mạnh vì k đo LLM (retrieval-only).

---

### Bước 5: Reranking
**File đích:** `home_works/aie10/pipeline_v3_rerank.py`

Cần làm:
- [ ] Thêm CrossEncoderReranker với model BAAI/bge-reranker-v2-m3
- [ ] ContextualCompressionRetriever: hybrid k=15 -> rerank top_n=5
- [ ] Chạy evaluation, ghi kết quả `evaluation_aie10_v3_rerank.json`

**Kết quả:**
| Metric | Value | Δ vs Hybrid |
|--------|-------|:------------:|
| Context Recall | | |
| Keyword Coverage | | |
| Avg Latency | | |

**Ghi chú:**

---

### Bước 6: Bảng so sánh tổng hợp
**File đích:** `home_works/aie10/comparison.md`

Cần làm:
- [ ] Tổng hợp 4 bảng kết quả
- [ ] Tính Δ accumulation từ baseline
- [ ] Nhận xét từng kỹ thuật

### Bước 7: Kết luận
Cần làm:
- [ ] Kỹ thuật nào giúp nhiều nhất
- [ ] Recommend cho dữ liệu này

---

## Kết quả cuối cùng

| Technique | Context Recall | Keyword Coverage | Avg Latency (s) |
|-----------|:-------------:|:----------------:|:----------------:|
| **Baseline** (chunk=1000/200, vector k=4) | 84.62% | 92.22% | 10.78 |
| **+ Chunking** (chunk=512/128, structure-aware) | 84.62% | 93.33% | 11.05 |
| **+ Hybrid** (BM25+Vector, k=15) | **92.31%** | *~93%* | *~10* |

## Kết luận
- **Hybrid Search** là kỹ thuật có impact lớn nhất: +7.69% Context Recall
- Chunking improved giúp nhẹ (+1.11% KW) nhưng pure vector search không đủ để xử lý semantic gap
- Cần reranking (cross-encoder) để lọc top 5 từ danh sách k=15 — chưa implement do API rate limit
- Hướng cải thiện thêm: query expansion (thêm từ đồng nghĩa "treo"="đơ"="khởi động lại") để xử lý cases như #7
