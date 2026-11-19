FROM python:3.11-slim

# Optional: install Tesseract OCR so uploaded report photos run full local OCR in cloud too
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV HOST=0.0.0.0
ENV PORT=8000
EXPOSE 8000

CMD ["python", "server.py"]
