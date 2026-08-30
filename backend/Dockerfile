FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/backend

COPY backend/requirements.txt /app/backend/requirements.txt

# Install CPU-only PyTorch first.
RUN pip install --no-cache-dir \
    torch==2.12.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install the remaining Python dependencies.
RUN pip install --no-cache-dir \
    -r /app/backend/requirements.txt \
    --extra-index-url https://pypi.org/simple

COPY backend /app/backend

RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]