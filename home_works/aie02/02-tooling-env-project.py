### tooling và môi trường dự án python

## 1. Vì sao cần môi trường ảo (virtual environment) cho dự án python?

# python có venv để isolate môi trường theo project => python -m venv .venv
# 3 thứ cần insolate theo dự án:
# - phiên bản python (uv lo cái này)
# - các package, thư viện (+ phiên bản) cài đặt (uv lo cái này)
# - các biến môi trường (env) và API KEY (.env lo cái này)

# => mỗi dự án nên có 1 môi trường ảo riêng, không dùng chung với các dự án khác
# => file  pyproject.toml là file cấu hình dự án, chứa thông tin về môi trường, package, phiên bản python, b+ file uv.lock là file lock để đảm bảo các package cài đặt đúng phiên bản, tránh xung đột giữa các dự án => cài trên máy nào cũng giống nhau

## 2. uv là gì? sử dụng ra sao?

# uv là trình quản lý package và môi trường ảo cho python, thay thế pip, venv, poetry, pipenv, conda, virtualenv, pyenv... (viết bằng Rust)
# pip chỉ cài package, venv chỉ tạo môi trường ảo, uv kết hợp cả 2: quản lý phiên bản python + tạo/ khóa env ảo.

# # cài uv (Linux / macOS)
# curl -LsSf https://astral.sh/
# uv/install.sh | sh
# # Windows (PowerShell)
# irm https://astral.sh/uv/
# install.ps1 | iex
# uv --version # kiểm tra đã cài

# => coi uv như 'trạm điều khiển' của dự án Python: từ tạo project, cài gói, tới chạy code đều qua nó

## 3. khởi tạo dự án với uv
# 3.1. uv init <tên dự án> # tạo project mới, tạo file pyproject.toml
# => uv init tạo sẵn bộ khung dự án chuẩn: file cấu hình, pin phiên bản Python,
# README, và khởi tạo git.

# uv init llm-app
# cd llm-app
# # cây thư mục sinh ra:
# llm-app/
# ├── .python-version (pin phiên bản python)
# ├── pyproject.toml (khai báo package, môi trường, script)
# ├── README.md (giới thiệu dự án)
# ├── main.py
# ├── uv.lock (lock phiên bản package sau khi uv add <package>)
# ├── .venv/ (môi trường ảo, chứa các package cài đặt)


# 3.2. thêm bớt thư viện (package) với uv
# uv add <package> # cài package vào dự án
# uv remove <package> # gỡ package khỏi dự án

# # thư viện chạy thật
# uv add anthropic pydantic httpx
# # thư viện chỉ dùng khi phát triển
# uv add --dev ruff ty pytest
# # gỡ một gói
# uv remove httpx

# 3.3. chạy code với uv
# uv run python main.py # chạy file main.py trong môi trường ảo

#  chạy, không cần 'source .venv/bin/activate'
# uv run python main.py
# uv run pytest

# 3.4. uv sync # đồng bộ môi trường ảo với file pyproject.toml và uv.lock
# dựng lại .venv theo đúng lock, cài đúng phiên bản package, tránh xung đột giữa các dự án
# uv -sync --frozen # trên CI/deploy: khớp lock, cài đúng phiên bản package, không update package mới

# # cài đúng theo lock (đồng đội / CI)
# uv sync
# uv sync --frozen # CI: khớp lock

## tổng kết: uv init (tạo project) => uv add/remove (cài/gỡ package) => uv run (chạy code), uv sync (đồng bộ môi trường ảo với lock)


### Project và khóa phụ thuộc

## 1. pyproject.toml là gì?
# file cấu hình dự án, chứa thông tin về môi trường, package, phiên bản python

# name = "llm-app"
# requires-python = ">=3.13"
# dependencies = [
# "anthropic>=0.40",
# "pydantic>=2.7",
# ]
# [dependency-groups]
# dev = ["ruff", "ty", "pytest"]

## 2. uv.lock & 3 file phải commit đúng

# uv.lock là khóa CHÍNH XÁC version của mọi package cho mọi nền tảng, để đảm bảo môi trường ảo của mọi người đều giống nhau, tránh xung đột giữa các dự án.

# quy tắc commit: pyproject.toml + uv.lock + .python-version

# # commit 3 file này:
# pyproject.toml ✓
# uv.lock ✓ (đừng sửa tay)
# .python-version ✓
# # .gitignore: 2 folder không commit:
# .venv/ # tái tạo bằng uv sync
# __pycache__/

# => lock file là thứ biến "package" (thứ chắc giống nhau) thành "phiên bản" (thứ chắc giống nhau) => tránh xung đột giữa các dự án, giữa các máy, giữa các môi trường. => đứng sửa tay uv.lock

## 3. bố cục 1 dự án python chuẩn
# hãy để trong src/<tên dự án> tất cả code của dự án, ví dụ src/llm_app, src/weather_app, src/my_project

# llm-app/
# ├── .python-version (pin phiên bản python)
# ├── pyproject.toml (khai báo package, môi trường, script)
# ├── uv.lock (lock phiên bản package sau khi uv add <package>)
# ├── .venv/ (môi trường ảo, chứa các package cài đặt)
# ├── src/ (chứa code dự án)
#       ├─── llm_app/ (code dự án)
#       ├─── __init__.py (khởi tạo package)
# ├── tests/ (chứa test code)

## 4. script 1 file
# với script lẻ trong 1 file .py, ta khai báo package cần ngay trong file

# /// script
# requires-python = ">=3.13"
# dependencies = ["httpx"]
# ///

# import httpx
# print(httpx.get(...))
# uv run script.py # tự cài httpx rồi chạy

## tóm lại: project được sync mà ko bị xung đột là nhờ:
# 1. pyproject.toml: khai báo package, môi trường, phiên bản python
# 2. uv.lock: lock phiên bản package sau khi uv add <package>
# 3. .python-version: pin phiên bản python

# => sync lại uv sync để dựng lại .venv theo đúng lock, cài đúng phiên bản package, tránh xung đột giữa các dự án


### Chất lượng code và test

## 1. ruff - format + lint code
# ruff là tool lint code, kiểm tra code có chuẩn style thay cho Flake8

# # định dạng lại toàn bộ code
# uv run ruff format .
# # soi lỗi & tự sửa những gì sửa được
# uv run ruff check --fix
# # cấu hình trong pyproject.toml:
# [tool.ruff] (cấu hình tại root dự án, trong file pyproject.toml)
# line-length = 100

# =>  trước khi commit, chạy 'ruff format' rồi 'ruff check --fix' — code luôn sạch và thống nhất, gần như miễn phí thời gian.

## 2. typecheck với ty
# ty là tool typecheck code, kiểm tra type hint thay cho mypy

# Dựa vào type hint (Topic 1) để bắt lỗi KIỂU trước khi chạy: truyền sai kiểu,
# quên None, sai tên trường Pydantic… hiện ngay trong editor

# # ty — nhanh, của Astral (beta) (nhanh)
# uv add --dev ty
# uv run ty check
# # hoặc mypy — ổn định, phổ biến (chậm)
# uv add --dev mypy

# => trước khi commit, chạy 'ty check' — code luôn đúng kiểu, gần như miễn phí thời gian.

## 3. test với pytest
# pytest là tool test code, kiểm tra code có đúng logic thay cho unittest

# uv run pytest # chạy tất cả test trong thư mục tests/

# llm-app/
# ├── .python-version (pin phiên bản python)
# ├── pyproject.toml (khai báo package, môi trường, script)
# ├── uv.lock (lock phiên bản package sau khi uv add <package>)
# ├── .venv/ (môi trường ảo, chứa các package cài đặt)
# ├── src/ (chứa code dự án)
#       ├─── llm_app/ (code dự án)
#       ├─── __init__.py (khởi tạo package)
# ├── tests/ (chứa test code) (chạy pytest các test trong thư mục này)


### 4. CI/CD và pre-commit

## 4.1. tạo file pre-commit-config.yaml trong root dự án, để chạy ruff, ty, pytest trước khi commit code lên git
# => Chạy format + lint + type check TỰ ĐỘNG mỗi lần git commit. Code lỗi thì
# commit bị chặn — giữ nhánh chính luôn sạch

# trong pre-commit-config.yaml:
# fail_fast: true (dừng commit nếu có lỗi)

# repos:
#   - repo: local (local repo)
#     hooks: (đưa vào các hook cần chạy trước khi commit)
#       - id: ruff-format
#         name: ruff format
#         entry: uv run ruff format --check
#         language: system
#         types_or: [python, pyi]

#       - id: ruff-check
#         name: ruff lint
#         entry: uv run ruff check
#         language: system
#         types_or: [python, pyi]

#       - id: ty-check
#         name: ty type check
#         entry: uv run ty check
#         language: system
#         pass_filenames: false


# cài pre-commit: uv add --dev pre-commit (add pre-commit vào dev dependencies)
# cài hook: uv run pre-commit install

## 4.2. CI/CD (GitHub Actions - giống pre-commit nhưng chạy trên server)

# CI ta cấu hình trong .github/workflows/ci.yml, để chạy ruff, ty, pytest trên server mỗi lần push code lên git, hoặc khi tạo pull request. Nếu code lỗi thì CI bị fail — giữ nhánh chính luôn sạch
# CD ta cấu hình trong .github/workflows/cd.yml, để deploy code lên server mỗi lần merge code vào nhánh chính (main/master). Nếu code lỗi thì CD bị fail — tránh deploy code lỗi lên server

## trong ci.yml:

# example sau đang thực hiện cài mội trường ảo, cài package, chạy ruff, ty, pytest trên server mỗi lần push code lên git, hoặc khi tạo pull request. Nếu code lỗi thì CI bị fail

# name: CI

# on:
#   push:
#     branches: [main, dev]
#   pull_request:
#     branches: [main, dev]

# jobs:
#   lint:
#     runs-on: ubuntu-latest
#     steps:
#       - uses: actions/checkout@v4

#       - name: Install uv
#         uses: astral-sh/setup-uv@v5

#       - name: Set up Python
#         uses: actions/setup-python@v5
#         with:
#           python-version-file: ".python-version"

#       - name: Install dependencies
#         run: uv sync --locked --all-extras --dev

#       - name: Lint with ruff
#         run: uv run ruff check --diff

#       - name: Check formatting with ruff
#         run: uv run ruff format --check --diff

#       - name: Check types with ty
#         run: uv run ty check

#       - name: run tests with pytest
#         run: uv run pytest


## trong cd.yml:

# cd ta sẽ thực hiện deploy code lên server mỗi lần merge code vào nhánh chính (main/master). Nếu code lỗi thì CD bị fail — tránh deploy code lỗi lên server

# example sau đang thực hiện cài mội trường ảo, cài package, chạy ruff, ty, pytest trên server mỗi lần merge code vào nhánh chính (main/master). Nếu code lỗi thì CD bị fail — tránh deploy code lỗi lên server (hoặc build docker image, hoặc deploy lên server, hoặc push lên pypi...)

# name: CD

# on:
#   push:
#     branches: [main]

# jobs:
#   # Job 1: Chạy kiểm tra code (Lint, Format, Type Check, Test)
#   test:
#     runs-on: ubuntu-latest
#     steps:
#       - uses: actions/checkout@v4

#       - name: Install uv
#         uses: astral-sh/setup-uv@v5

#       - name: Set up Python
#         uses: actions/setup-python@v5
#         with:
#           python-version-file: ".python-version"

#       - name: Install dependencies
#         run: uv sync --locked --all-extras --dev

#       - name: Lint with ruff
#         run: uv run ruff check --diff

#       - name: Check formatting with ruff
#         run: uv run ruff format --check --diff

#       - name: Check types with ty
#         run: uv run ty check

#       - name: Run tests with pytest
#         run: uv run pytest

#   # Job 2: Build Docker Image và Deploy lên Server bằng SSH
#   deploy:
#     needs: test # Bắt buộc Job test phải Pass thì mới chạy Job deploy
#     runs-on: ubuntu-latest

#     steps:
#       - uses: actions/checkout@v4

#       # 1. Đăng nhập vào Docker Hub
#       - name: Log in to Docker Hub
#         uses: docker/login-action@v3
#         with:
#           username: ${{ secrets.DOCKERHUB_USERNAME }}
#           password: ${{ secrets.DOCKERHUB_TOKEN }}

#       # 2. Build và Push Image lên Docker Hub
#       - name: Build and push Docker image
#         uses: docker/build-push-action@v5
#         with:
#           context: .
#           push: true
#           # Thay my-python-app bằng tên repository bạn đã tạo trên Docker Hub
#           tags: ${{ secrets.DOCKERHUB_USERNAME }}/my-python-app:latest

#       # 3. Kết nối SSH vào server và thực hiện các lệnh Deploy
#       - name: Deploy to server via SSH
#         uses: appleboy/ssh-action@v1.0.3
#         with:
#           host: ${{ secrets.SERVER_HOST }}
#           username: ${{ secrets.SERVER_USERNAME }}
#           key: ${{ secrets.SERVER_SSH_KEY }}
#           script: |
#             # Đăng nhập Docker Hub trên server (Bắt buộc nếu repository Docker Hub của bạn là Private)
#             echo "${{ secrets.DOCKERHUB_TOKEN }}" | docker login -u "${{ secrets.DOCKERHUB_USERNAME }}" --password-stdin

#             # Kéo image mới nhất từ Docker Hub về server
#             docker pull ${{ secrets.DOCKERHUB_USERNAME }}/my-python-app:latest

#             # Dừng và xóa container đang chạy (bỏ qua lỗi nếu container chưa tồn tại)
#             docker stop my-app-container || true
#             docker rm my-app-container || true

#             # Khởi tạo container mới từ image vừa kéo về
#             docker run -d \
#               --name my-app-container \
#               --restart unless-stopped \
#               -p 8000:8000 \
#               ${{ secrets.DOCKERHUB_USERNAME }}/my-python-app:latest

#             # Dọn dẹp các image cũ lơ lửng (dangling) để tránh đầy ổ cứng server
#             docker image prune -f


### Extra: 5. Cấu hình Secrets trên GitHub Repository để chạy CD
# Trên Docker Hub:

# Tạo một repository với tên tương ứng (ví dụ: my-python-app).

# Vào Account Settings -> Security -> New Access Token. Tạo một token có quyền Read/Write và copy mã token đó (Không nên dùng mật khẩu thật để đảm bảo bảo mật).

# Trên GitHub Repository (Settings -> Secrets and variables -> Actions):
# Bạn cần thêm 5 biến môi trường (Secrets) sau để cung cấp "nguyên liệu" cho file .yml:

# DOCKERHUB_USERNAME: Tên đăng nhập Docker Hub của bạn.

# DOCKERHUB_TOKEN: Mã Access Token vừa tạo ở trên.

# SERVER_HOST: IP Public của VPS/Server.

# SERVER_USERNAME: User dùng để SSH (ví dụ: root, ubuntu, ec2-user).

# SERVER_SSH_KEY: Nội dung Private Key để truy cập server (thường nằm ở ~/.ssh/id_rsa). Bạn phải đảm bảo Public Key tương ứng đã được thêm vào file ~/.ssh/authorized_keys trên server.


### 6. .env và biến môi trường tại local và trên server
## 1. .env là file chứa các biến môi trường (environment variables) và API KEY, để tránh hardcode trực # tiếp trong code, giúp bảo mật và dễ thay đổi cấu hình.

# tránh việc hardcode API key trong code, vì nếu lỡ commit lên git thì sẽ bị lộ, hoặc nếu muốn đổi # key thì phải sửa code. Thay vào đó, ta dùng biến môi trường (environment variables) và file .env # để lưu trữ các giá trị nhạy cảm này.

# trong .gitignore
# .env # tránh commit file .env lên git, vì chứa API key nhạy cảm

# ví dụ: trong file .env:
# GEMINI_API_KEY=your_gemini_api_key
# OPENAI_API_KEY=your_openai_api_key

## 2. tại local, ta có thể load các biến môi trường từ file .env bằng thư viện python-dotenv hoặc pydantic_settings.

# from pydantic_settings import BaseSettings
# class Settings(BaseSettings):
# anthropic_api_key: str
# model: str = "claude-..."
# model_config = {"env_file": ".env"}
# settings = Settings() # validate lúc chạy
# client = Anthropic(api_key=settings
# .anthropic_api_key)

# vì sao hơn os.getenv ?
# os.getenv trả chuỗi hoặc None, dễ quên kiểm tra. Settings validate sẵn, tự
# ép kiểu, gợi ý (autocomplete) — cấu hình sạch như dữ liệu
