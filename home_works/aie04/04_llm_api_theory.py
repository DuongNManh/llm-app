### nền tảng chung cho việc gọi LLM api
# gồm: message & role, tham số, đọc response, streaming, token & key


## Message & Role

# 1. một cuộc hội thoại với LLM là 1 danh sách các message, mỗi message có role và content
# model đọc cả list message để tạo ra response tiếp theo

# các role (vai trò):
# - system: vai trò của hệ thống, hướng dẫn model cách trả lời
# - user: vai trò của người dùng, câu hỏi hoặc yêu cầu từ người dùng
# - assistant: câu model đã trả lời (đưa lại để model nhớ hội thoại trước đó)

messages = [
    {"role": "system", "content": "Bạn là trợ lý ngắn gọn"},
    {"role": "user", "content": "Thủ đô Việt Nam?"},
]
# Claude tách 'system' ra tham số riêng,
# nhưng ý tưởng ba vai trò là như nhau.

# 2. Các tham số điều khiển
temperature = 0.7  # độ sáng tạo, 0-1, càng cao càng sáng tạo
# tác vụ cần ổn định (trích xuất, phân loại) -> temperature thấp
max_tokens = 100  # số token tối đa model trả về (chặn tràn, chặn tốn tiền)
# system / instructions = "Bạn là chuyên gia về ...." # đặt vai trò và quy tắc trả lời của model

# Nhớ: cần kết quả ổn định (trích xuất, phân loại) → temperature thấp. Cần đa dạng → cao. Luôn đặt max_tokens.
