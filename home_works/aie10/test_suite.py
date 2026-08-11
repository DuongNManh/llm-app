"""Bộ test 15 câu hỏi cho RAG Samsung - 4 categories."""

TEST_CASES = [
    # ─────────── GENERAL (4 câu) ───────────
    {
        "id": 1,
        "question": "Hãy hướng dẫn kết nối Wifi cho điện thoại Samsung.",
        "expected_answer_contains": ["Cài đặt", "Kết nối", "Wi-Fi", "mạng Wi-Fi"],
        "expected_source_pages": [65, 66],
        "category": "general",
    },
    {
        "id": 2,
        "question": "Cách chụp màn hình trên điện thoại Samsung?",
        "expected_answer_contains": ["phím Cạnh", "Giảm âm lượng", "cùng lúc"],
        "expected_source_pages": [30],
        "category": "general",
    },
    {
        "id": 3,
        "question": "Pin điện thoại Samsung nên sạc như thế nào cho đúng?",
        "expected_answer_contains": ["sạc", "pin", "USB Type-C"],
        "expected_source_pages": [13, 14],
        "category": "general",
    },
    {
        "id": 4,
        "question": "Làm thế nào để chuyển dữ liệu từ máy cũ sang máy Samsung mới?",
        "expected_answer_contains": ["Smart Switch", "dữ liệu", "chuyển"],
        "expected_source_pages": [21, 22],
        "category": "general",
    },
    # ─────────── SEMANTIC GAP (5 câu) ───────────
    {
        "id": 5,
        "question": "Làm sao để vào Internet qua Wifi trên máy Samsung?",
        "expected_answer_contains": ["Cài đặt", "Kết nối", "Wi-Fi"],
        "expected_source_pages": [65, 66],
        "category": "semantic_gap",
    },
    {
        "id": 6,
        "question": "Điện thoại Samsung bị treo logo, làm thế nào để khắc phục?",
        "expected_answer_contains": ["nhấn và giữ", "phím Cạnh", "phím Giảm âm lượng", "7 giây", "khởi động lại"],
        "expected_source_pages": [19],
        "category": "semantic_gap",
    },
    {
        "id": 7,
        "question": "Máy Samsung của tôi bị đơ, không bấm được gì, phải làm sao?",
        "expected_answer_contains": ["nhấn và giữ", "phím Cạnh", "phím Giảm âm lượng", "khởi động lại"],
        "expected_source_pages": [19],
        "category": "semantic_gap",
    },
    {
        "id": 8,
        "question": "Samsung của tôi bị nóng quá, có sao không?",
        "expected_answer_contains": ["nóng", "thiết bị", "sạc", "ứng dụng"],
        "expected_source_pages": [6, 7, 8],
        "category": "semantic_gap",
    },
    {
        "id": 9,
        "question": "Làm cách nào sao chép ảnh từ Samsung qua máy vi tính?",
        "expected_answer_contains": ["Smart Switch", "máy tính", "dữ liệu"],
        "expected_source_pages": [22],
        "category": "semantic_gap",
    },
    # ─────────── CODE / MODEL NAME (4 câu) ───────────
    {
        "id": 10,
        "question": "SM-A125F/DS dùng loại thẻ SIM nào?",
        "expected_answer_contains": ["nano SIM", "SIM"],
        "expected_source_pages": [15, 16],
        "category": "code_model",
    },
    {
        "id": 11,
        "question": "Điện thoại Samsung có hỗ trợ Dolby Atmos không?",
        "expected_answer_contains": ["Dolby Atmos", "âm thanh", "Cài đặt"],
        "expected_source_pages": [72],
        "category": "code_model",
    },
    {
        "id": 12,
        "question": "Smart Switch có thể chuyển dữ liệu bằng cách nào?",
        "expected_answer_contains": ["Smart Switch", "Không dây", "máy tính", "dữ liệu"],
        "expected_source_pages": [21, 22],
        "category": "code_model",
    },
    {
        "id": 13,
        "question": "Samsung Members giúp ích gì khi máy gặp vấn đề?",
        "expected_answer_contains": ["Samsung Members", "hỗ trợ", "chẩn đoán"],
        "expected_source_pages": [56],
        "category": "code_model",
    },
    # ─────────── OUT OF SCOPE (2 câu) ───────────
    {
        "id": 14,
        "question": "Giá bán của sản phẩm này là bao nhiêu?",
        "expected_answer_contains": ["Tài liệu không đề cập", "không đề cập"],
        "expected_source_pages": [],
        "category": "out_of_scope",
    },
    {
        "id": 15,
        "question": "Bảo hành điện thoại Samsung bao lâu?",
        "expected_answer_contains": ["Tài liệu không đề cập", "không đề cập"],
        "expected_source_pages": [],
        "category": "out_of_scope",
    },
]


def get_test_cases(category=None):
    if category is None:
        return TEST_CASES
    return [tc for tc in TEST_CASES if tc["category"] == category]


def print_test_summary():
    from collections import Counter
    cats = Counter(tc["category"] for tc in TEST_CASES)
    print(f"Tổng số test cases: {len(TEST_CASES)}")
    for cat, count in cats.items():
        print(f"  - {cat}: {count}")
