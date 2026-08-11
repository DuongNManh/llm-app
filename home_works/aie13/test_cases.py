# ---------------- các ca kiểm thử ----------------
CASES = [
    {
        "id": 1,
        "name": "Ca 1 - Hỏi đúng nội dung CÓ trong tài liệu (kèm nguồn, trang)",
        "queries": [
            ("thread_1", "Hướng dẫn sử dụng Wi-Fi trên điện thoại Samsung như thế nào?"),
        ],
        "check": "in_doc",
    },
    {
        "id": 2,
        "name": "Ca 2 - Hỏi điều KHÔNG có trong tài liệu (không bịa)",
        "queries": [
            ("thread_2", "Cách nấu phở bò ngon nhất là gì?"),
        ],
        "check": "out_doc",
    },
    {
        "id": 3,
        "name": "Ca 3 - Số liệu trong tài liệu: tra kho RỒI dùng calculator",
        "queries": [
            ("thread_3",
             "SM-A125F có dung lượng pin là bao nhiêu? Nếu dùng 2 cục pin như vậy thì tổng là bao nhiêu?"),
        ],
        "check": "math",
    },
    {
        "id": 4,
        "name": "Ca 4 - Hỏi nối đa lượt (đại từ) + /new thì quên",
        "queries": [
            ("thread_4", "Cách chụp ảnh màn hình trên điện thoại Samsung?"),
            ("thread_4", "Vậy để bật tính năng đó thì vào phần nào trên máy?"),
            ("thread_5", "Vậy để bật tính năng đó thì vào phần nào trên máy?"),  # sau /new -> phải quên
        ],
        "check": "multi_turn",
    },
    {
        "id": 5,
        "name": "Ca 5 - Giao tiếp thường (KHÔNG được gọi tool)",
        "queries": [
            ("thread_6", "Xin chào!"),
        ],
        "check": "greeting",
    },
]