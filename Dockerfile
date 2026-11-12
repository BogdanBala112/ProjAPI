FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.27 /uv /uvx /bin/

WORKDIR /app


COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project


COPY main.py llm_call.py ./

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
