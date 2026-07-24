# Architecture - RAG Pipeline

## Tổng quan

Single-file RAG pipeline (`index.py`) với 2 pha riêng biệt:

| Phase | Khi chạy | Chức năng |
|-------|----------|-----------|
| **Indexing** | 1 lần (offline) | PDF → chunks → vector store |
| **Query** | Nhiều lần (online) | question → retrieve → augment → generate |

## Flow

```
                     INDEXING PHA
PDF ──> load_pdf() ──> split_documents() ──> create_vectorstore() ──> chroma_db/
  PyPDFLoader        RecursiveCharacter     GoogleGenerativeAI        Chroma (lưu bền)
                     TextSplitter           Embeddings (batch + sleep)

                     QUERY PHA
Question ──> as_retriever(k=4) ──> format_docs() ──> prompt ──> llm ──> answer
  input         Chroma               [Trang X] ...     LCEL      Gemini
              similarity search                                  1.5-flash
```

## Thành phần chính

### 1. `load_pdf(path)` — Load PDF
- **Input**: đường dẫn file PDF
- **Output**: `list[Document]` (langchain Document)
- Mỗi Document chứa `page_content` (text) + `metadata` (page number, source)
- Dùng `PyPDFLoader` — tách từng trang thành Document riêng

### 2. `split_documents(docs, chunk_size, chunk_overlap)` — Split
- **Input**: `list[Document]` (mỗi Document = 1 trang)
- **Output**: `list[Document]` (chunk nhỏ)
- Dùng `RecursiveCharacterTextSplitter`:
  - Chia theo `["\n\n", "\n", " ", ""]` (từ lớn → nhỏ)
  - `chunk_size=1000` ký tự
  - `chunk_overlap=200` — giữa 2 chunk liền kề overlap 200 ký tự
  - `add_start_index=True` — ghi lại vị trí ký tự gốc trong tài liệu

### 3. `create_embeddings()` — Embedding Function
- **Model**: `gemini-embedding-001` (768-dim vector)
- **Wrapper**: `GoogleGenerativeAIEmbeddings` từ `langchain_google_genai`

### 4. `create_vectorstore(documents, persist_dir)` — Embed + Store
- Xóa thư mục `persist_dir` nếu tồn tại (re-index sạch)
- Tạo `Chroma` collection rỗng với `embedding_function`
- **Rate limiting**: chia documents thành batch 10, sleep 10s giữa batch
- Lý do: Gemini free tier chỉ cho 100 RPM, ~30k TPM
- **Kết quả**: vector database tại `./chroma_db` — có thể load lại mà không cần re-embed

### 5. `load_vectorstore(persist_dir)` — Load DB đã lưu
- **Không embed lại** — chỉ load index đã lưu từ disk
- Dùng cùng `embedding_function` (cùng model) để query

### 6. `build_rag_chain(vectorstore, k)` — LCEL Chain

```
             User input (string)
                    │
                    ▼
     ┌──────────────────────────────┐
     │  {"context": ...,            │
     │   "question": input}         │
     └──────┬──────────────┬────────┘
            │              │
            ▼              ▼
     retriever       RunnablePassthrough
     (Chroma         (pass through
      similarity      question as-is)
      search k=4)
            │              │
            ▼              │
     format_docs()         │
     [Trang X] ...         │
            │              │
            └──────┬───────┘
                   ▼
          ChatPromptTemplate
          system: SYSTEM_PROMPT
          human:  "NGỮ CẢNH:\n{context}\n\nCâu hỏi: {question}"
                   │
                   ▼
          ChatGoogleGenerativeAI
          model="gemini-1.5-flash"
          temperature=0
                   │
                   ▼
          StrOutputParser
          (trích text từ LLM response)
                   │
                   ▼
                answer
```

### 7. `format_docs(docs)` — Format
- Kết hợp các chunk retrieved thành string
- Format: `[Trang X] nội dung chunk\n\n---\n\n[Trang Y] nội dung chunk...`
- Page number từ `doc.metadata["page"]` (do PyPDFLoader tự thêm)

### 8. `ask(chain, question)` — Hỏi
- `chain.invoke(question)` — chạy full LCEL pipeline

## Xử lý rate limit Gemini

| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| `batch_size` | 10 chunks/batch | Tránh vượt 100 RPM |
| `sleep` | 10 giây | ~60 RPM thực tế (safe margin) |
| `requests_per_day` | 1000 (API limit) | 197 chunks / 10 = 20 requests → OK |
| Query phase | 1 request | Không bị limit |

## Prompt grounding

`SYSTEM_PROMPT` (hằng số) gồm 4 yêu cầu:
1. Chỉ dựa vào ngữ cảnh được cung cấp
2. Nếu không có → "Tài liệu không đề cập đến vấn đề này."
3. Trả lời kèm trích dẫn
4. Trả lời bằng tiếng Việt

Kết hợp với `temperature=0` giúp model bám sát tài liệu, giảm hallucination.

## Các constants

| Constant | Giá trị | Mục đích |
|----------|---------|----------|
| `PDF_PATH` | `documents/huongdansudungSamSung.pdf` | File PDF đầu vào |
| `CHROMA_DIR` | `./chroma_db` | Thư mục lưu vector store |
| `EMBED_BATCH_SIZE` | 10 | Batch cho rate limit |
| `EMBED_BATCH_SLEEP` | 10s | Sleep giữa batch |
| `RETRIEVER_K` | 4 | Số chunks retrieved |
