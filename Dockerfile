FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./app/requirements.txt /tmp/requirements.txt

# ---- 핵심 구간 ----
RUN pip install --upgrade pip==24.2 && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir "numpy<2" --force-reinstall && \
    python -c "import numpy; print('✅ NumPy version:', __import__('numpy').__version__)"
# -------------------

EXPOSE 7878
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7878", "--reload"]
