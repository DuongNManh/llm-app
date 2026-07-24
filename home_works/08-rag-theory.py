### RAG là gì? và tại sao cần RAG?

# vấn đề: LLM không biết dữ liệu của BẠN
# chỉ biết những gì đã học được từ dữ liệu huấn luyện, không biết dữ liệu riêng, mới
# => dễ bịa khi thiếu thông tin (hallucination)
# example: 
# 1) Why hotel are expensive in the weekend? => LLM: Because of high demand in the weekend.
# 2) Why Muong Thanh hotal are very expensive this weekend? => LLM không biết => bịa liền.


## 1. RAG: Retrieval Augmented Generation
# tra (retrieval) cứu dữ liệu riêng của bạn -> đưa vào prompt cho LLM (augment) -> model trả lời (generation) dựa trên tài liệu đó.

# RAG giải quyết vấn đề gì?
# => Dùng dữ liệu riêng của bạn (nội bộ, db, tri thức sản phẩm mới) - thứ model chưa từng thấy khi huấn luyện
# => cập nhật dữ liệu mới (new knowledge) - dễ dàng, không cần huấn luyện lại model (tốn kém)
# => giảm bịa (hallucination) - model trả lời dựa trên dữ liệu riêng của bạn, không bịa.
# => trích dẫn được nguồn (source) - biết được câu trả lời đến từ tài liệu/ đoạn nào


## 2. sơ đồ tổng quan 2 pha của RAG:

# 1.indexing: làm 1 lần (offline) - đưa dữ liệu riêng của bạn vào vector database (vector embedding)
# - tài liệu (text, pdf, web,...) -> chunking -> embedding -> vector database

# 2. query: làm mỗi lần có câu hỏi (online) - tìm kiếm dữ liệu riêng của bạn trong vector database -> đưa vào prompt cho LLM -> model trả lời dựa trên tài liệu đó.
# câu hỏi -> embedding -> vector database -> tìm các đoạn (chunk) tương tự nhất -> đưa vào prompt cho LLM -> model trả lời dựa trên tài liệu đó.