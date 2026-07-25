FROM runpod/pytorch:2.5.1-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    diffusers==0.29.2 transformers==4.27.3 accelerate==0.31.0 \
    Pillow numpy opencv-python-headless einops scipy scikit-image \
    tqdm PyYAML xformers ninja \
    fastapi uvicorn httpx

RUN git clone https://github.com/Zheng-Chong/CatVTON.git /catvton

COPY rp_handler.py .

CMD ["python3", "-u", "rp_handler.py"]
