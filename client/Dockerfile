FROM python:3.10-slim

WORKDIR /app

# libgomp1 requis par PyTorch pour le parallélisme CPU
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# torch CPU-only (~200MB au lieu de ~2GB pour la version CUDA)
RUN pip install --no-cache-dir \
    flwr==1.5.0 \
    torch==2.2.0+cpu \
    pandas \
    pyarrow \
    scikit-learn \
    numpy \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY client.py privacy.py ./
COPY models/ ./models/

CMD ["python", "client.py"]
