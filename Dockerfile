FROM runpod/pytorch:2.5.1-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /

RUN git clone https://github.com/Zheng-Chong/CatVTON.git /catvton

COPY builder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/handler.py .

CMD ["python", "-u", "/handler.py"]
