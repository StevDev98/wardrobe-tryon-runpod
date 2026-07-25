FROM runpod/pytorch:2.5.1-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/Zheng-Chong/CatVTON.git /catvton

RUN mkdir -p /weights && \
    python3 -c "
import sys; sys.path.insert(0, '/catvton')
from model.pipeline import CatVTONPipeline
import torch
pipe = CatVTONPipeline(
    base_ckpt='booksforcharlie/stable-diffusion-inpainting',
    attn_ckpt='zhengchong/CatVTON',
    attn_ckpt_version='mix',
    weight_dtype=torch.float16,
    device='cuda',
)
print('Model downloaded and cached')
"

COPY rp_handler.py .

CMD ["python3", "-u", "rp_handler.py"]
