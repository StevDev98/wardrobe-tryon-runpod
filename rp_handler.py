"""
RunPod Serverless handler for CatVTON Virtual Try-On
"""
import io
import base64
import traceback

import runpod

pipe = None


def load_model():
    global pipe
    if pipe is not None:
        return pipe

    import sys
    sys.path.insert(0, "/catvton")
    import torch

    from model.pipeline import CatVTONPipeline

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


def handler(job):
    try:
        job_input = job["input"]

        person_b64 = job_input["person_image"]
        garment_b64 = job_input["garment_image"]
        steps = job_input.get("num_inference_steps", 50)
        guidance = job_input.get("guidance_scale", 2.5)
        seed = job_input.get("seed", 42)

        from PIL import Image
        person_img = Image.open(io.BytesIO(base64.b64decode(person_b64))).convert("RGB")
        garment_img = Image.open(io.BytesIO(base64.b64decode(garment_b64))).convert("RGB")

        from utils import resize_and_crop, resize_and_padding
        person_img = resize_and_crop(person_img, (768, 1024))
        garment_img = resize_and_padding(garment_img, (768, 1024))

        import torch
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
        return {"image": base64.b64encode(buf.getvalue()).decode()}

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    load_model()
    runpod.serverless.start({"handler": handler})
