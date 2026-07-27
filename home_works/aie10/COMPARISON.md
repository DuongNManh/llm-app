# Bảng so sánh trước/sau — Cải thiện RAG

## Kết quả tổng hợp

| Kỹ thuật | Context Recall | Keyword Coverage | Avg Latency (s) | Δ Recall | Δ Coverage |
|----------|:-------------:|:----------------:|:----------------:|:--------:|:----------:|
| **1. Baseline** (chunk=1000/200, vector k=4) | 84.62% | 92.22% | 10.78 | — | — |
| **2. + Chunking** (chunk=512/128, structure-aware) | 84.62% | 93.33% | 11.05 | 0% | +1.11% |
| **3. + Hybrid** (BM25+Vector, k=15) | 92.31% | *~93% (est.)* | ~10 (est.) | **+7.69%** | — |

## Chi tiết per-test-case

| # | Category | Question | Baseline CR | Chunking CR | Hybrid CR |
|---|----------|----------|:-----------:|:-----------:|:---------:|
| 1 | general | Hướng dẫn kết nối Wifi | ✓ | ✓ | ✓ |
| 2 | general | Cách chụp màn hình | ✓ | ✓ | ✓ |
| 3 | general | Sạc pin đúng cách | ✓ | ✓ | ✓ |
| 4 | general | Chuyển dữ liệu từ máy cũ | ✓ | ✓ | ✓ |
| 5 | semantic_gap | Vào Internet qua Wifi | ✓ | ✓ | ✓ |
| 6 | semantic_gap | **Bị treo logo** | **✗** | **✗** | ✓ |
| 7 | semantic_gap | **Bị đơ, không bấm được** | **✗** | **✗** | **✗** |
| 8 | semantic_gap | Bị nóng quá | ✓ | ✓ | ✓ |
| 9 | semantic_gap | Sao chép ảnh qua máy tính | ✓ | ✓ | ✓ |
| 10 | code_model | SM-A125F/DS dùng SIM gì | ✓ | ✓ | ✓ |
| 11 | code_model | Dolby Atmos | ✓ | ✓ | ✓ |
| 12 | code_model | Smart Switch cách chuyển | ✓ | ✓ | ✓ |
| 13 | code_model | Samsung Members | ✓ | ✓ | ✓ |
| 14 | out_of_scope | Giá bán | ✓ | ✓ | ✓ |
| 15 | out_of_scope | Bảo hành | ✓ | ✓ | ✓ |

## Phân tích từng kỹ thuật

### Kỹ thuật 1: Chunking cải thiện (512/128, structure-aware)
- **Tác động**: Keyword Coverage +1.11%, Context Recall không đổi
- **Lý do**: Chunk nhỏ hơn (512 vs 1000) giúp mỗi chunk tập trung vào 1 ý, tăng precision. Tuy nhiên pure vector search vẫn không khắc phục được semantic gap — các câu hỏi diễn đạt khác vẫn không match được với chunk đúng.
- **Kết luận**: Cải thiện nhẹ, cần kết hợp với kỹ thuật khác.

### Kỹ thuật 2: Hybrid Search (BM25 + Vector, k=15)
- **Tác động**: Context Recall +7.69% (từ 84.62% → 92.31%)
- **Lý do**: BM25 bắt được keyword "treo", "logo", "khởi động" mà vector search bỏ lỡ. Với k=15, hybrid lấy rộng hơn nên page 19 (Buộc khởi động lại) lọt vào danh sách ứng viên.
- **#6 (treo logo) được fix**: Query có "treo" → BM25 match với page 19 có "bị treo"
- **#7 (bị đơ) vẫn fail**: Query "bị đơ, không bấm được gì" không có keyword chung với page 19

## Kết luận

1. **Kỹ thuật giúp nhiều nhất: Hybrid Search** — tăng Context Recall từ 84.62% → 92.31% (+7.69%)
2. **Chunking improved** giúp nhẹ về keyword coverage (+1.11%) nhưng không đáng kể
3. **Hybrid cần được bổ sung reranking** để lọc top kết quả chất lượng nhất từ danh sách ứng viên rộng (k=15 → top_n=5)
4. **Điểm yếu còn lại**: Câu hỏi không có keyword chung với tài liệu (#7) vẫn fail — cần thêm query expansion hoặc knowledge graph để giải quyết triệt để

### Đề xuất

| Hạng | Kỹ thuật | Impact | Cost | Priority |
|:----:|----------|:------:|:----:|:--------:|
| 1 | Hybrid (BM25+Vector) | **Cao** (+7.69% CR) | Thấp | **Làm ngay** |
| 2 | Rerank (Cross-encoder) | Trung bình (lọc top 5) | Cao | Khi có budget |
| 3 | Chunking improved | Thấp (+1.11% KW) | Rất thấp | Luôn áp dụng |
