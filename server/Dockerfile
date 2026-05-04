FROM python:3.10-slim

WORKDIR /app

# torch CPU-only (~200MB au lieu de ~2GB pour la version CUDA)
RUN pip install --no-cache-dir \
    flwr==1.5.0 \
    pandas \
    pyarrow \
    scikit-learn \
    numpy \
    torch==2.2.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY server.py .
COPY models/ ./models/

RUN mkdir -p /app/results /app/data

EXPOSE 8080

CMD ["python", "server.py"]