# Tracking - RAG Pipeline

## Step 1: PDF Loading
- [x] Sử dụng `PyPDFLoader` từ `langchain_community`
- [x] Đọc file `documents/huongdansudungSamSung.pdf`

## Step 2: Text Splitting
- [x] `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`
- [x] `add_start_index=True` để ghi nhớ vị trí gốc

## Step 3: Embedding
- [x] `GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")`
- [x] Rate limiting: batch_size=10, sleep=10s giữa các batch

## Step 4: Vector Store (Chroma)
- [x] `Chroma(persist_directory="./chroma_db")`
- [x] Xóa DB cũ khi re-index (dùng `shutil.rmtree`)

## Step 5: Retriever + LCEL Chain
- [x] `as_retriever(k=4)` — top 4 chunks tương đồng nhất
- [x] `ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)`
- [x] `ChatPromptTemplate` với grounding instruction (`SYSTEM_PROMPT`)
- [x] LCEL: `{"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()`

## Step 6: Query & Answer
- [x] `ask(chain, question)` — invoke chain
- [x] `format_docs()` — hiển thị `[Trang X]` + nội dung chunk
- [x] `test_queries()` — chạy 3 câu: trong tài liệu / diễn đạt khác / ngoài tài liệu

## Step 7: Grounding (không biết → nói không biết)
- [x] System prompt yêu cầu "Tài liệu không đề cập đến vấn đề này."
- [x] `temperature=0` để giảm hallucination

## TODO
- [ ] Tách `index.py` / `ask.py` (yêu cầu đề bài 2 file riêng)
- [ ] Thêm similarity threshold để filter chunks irrelevant
- [ ] In kèm similarity score trong output
- [ ] Xử lý khi chroma_db chưa tồn tại (load thất bại → báo lỗi rõ)
- [ ] CI test với pytest
