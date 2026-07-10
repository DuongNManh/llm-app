# 02-tooling-env

## Pre-commit

Cấu hình này chặn `git commit` nếu format, lint, hoặc type check có lỗi.

### Cài đặt một lần

```bash
uv sync --dev
uv run pre-commit install
```

### Kiểm tra toàn bộ repo

```bash
uv run pre-commit run --all-files
```

### Các hook đang chạy khi commit

- `ruff format --check`
- `ruff check`
- `ty check`
