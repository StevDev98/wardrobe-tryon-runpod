"""
RunPod Serverless worker for CatVTON — FastAPI endpoint
"""
import io
import base64
import os
import sys
import traceback

import torch
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

sys.path.insert(0, "/catvton")

app = FastAPI()

from model.pipeline import CatVTONPipeline
from utils import resize_and_crop, resize_and_padding

pipe = None


def load_model():
    global pipe
    if pipe is not None:
        return pipe
    print(f"Loading CatVTON on {torch.cuda.get_device_name()}")
    pipe = CatVTONPipeline(
        base_ckpt="booksforcharlie/stable-diffusion-inpainting",
        attn_ckpt="zhengchong/CatVTON",
        attn_ckpt_version="mix",
        weight_dtype=torch.float16,
        device="cuda",
    )
    print("CatVTON ready")
    return pipe


@app.on_event("startup")
async def startup():
    load_model()


@app.post("/runsync")
async def runsync(request: Request):
    try:
        body = await request.json()
        inp = body.get("input", body)

        person_b64 = inp["person_image"]
        garment_b64 = inp["garment_image"]
        steps = inp.get("num_inference_steps", 50)
        guidance = inp.get("guidance_scale", 2.5)
        seed = inp.get("seed", 42)

        person_img = Image.open(io.BytesIO(base64.b64decode(person_b64))).convert("RGB")
        garment_img = Image.open(io.BytesIO(base64.b64decode(garment_b64))).convert("RGB")

        person_img = resize_and_crop(person_img, (768, 1024))
        garment_img = resize_and_padding(garment_img, (768, 1024))

        gen = torch.Generator(device="cuda").manual_seed(seed)

        model = load_model()
        result = model(
            image=person_img,
            condition_image=garment_img,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=1024, width=768,
            generator=gen,
        )
        result_img = result[0] if isinstance(result, (list, tuple)) else result

        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        return {"output": {"image": base64.b64encode(buf.getvalue()).decode()}}

    except Exception:
        return JSONResponse(
            {"error": traceback.format_exc()}, status_code=500
        )


@app.get("/health")
async def health():
    return {"status": "ok", "gpu": torch.cuda.get_device_name()}


if __name__ == "__main__":
    import uvicorn
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)
