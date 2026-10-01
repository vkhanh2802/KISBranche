import os
import re
import gc

# CUDA_LAUNCH_BLOCKING=1 chỉ dùng khi debug lỗi CUDA, sẽ làm chậm mọi
# lệnh gọi kernel vì ép chạy đồng bộ. Bỏ khi chạy thật để tận dụng
# async kernel launch của PyTorch.
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Giảm phân mảnh VRAM khi generate nhiều lần liên tiếp trên GPU nhỏ.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

import torch
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
MIN_PIXELS = 256 * 256
MAX_PIXELS = 512 * 512


class VLMReranker:

    def __init__(self, use_4bit: bool = True):
        print("[VLM] Loading model...")

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        print("[VLM] Device:", self.device)

        if self.device.type == "cuda":
            print("[VLM] GPU:", torch.cuda.get_device_name(0))
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[VLM] Total VRAM: {total_vram:.1f} GB")

        # -----------------------------
        # Processor (giới hạn resolution ảnh)
        # -----------------------------
        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )

        # -----------------------------
        # Model
        # -----------------------------
        # RTX 3050 6GB: fp16 3B model đã chiếm gần hết VRAM, gần như
        # chắc chắn OOM khi cộng thêm ảnh + KV cache. Dùng 4-bit
        # quantization (bitsandbytes) để đưa weight về ~2.5-3GB,
        # để lại nhiều chỗ trống cho phần generate.
        if use_4bit and self.device.type == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_config,
                device_map={"": 0},
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            )
        else:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                torch_dtype=dtype,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            )
            self.model.to(self.device)

        self.model.eval()
        print("[VLM] Model loaded")

    @torch.inference_mode()
    def score(self, image_path, query, caption=None):
        image = Image.open(image_path).convert("RGB")

        if caption is None:
            caption = ""

        prompt = f"""You are a video retrieval reranker.

Query:
{query}

Caption:
{caption}

Evaluate how well this image matches the query using these criteria:
1. Subject match: Does the image contain the main object/person/action mentioned in the query?
2. Context match: Does the scene, setting, or activity align with the query's context?
3. Specificity: If the query mentions specific details (colors, counts, actions, locations), are they visibly present?
4. Caption consistency: Does the caption (if any) support or contradict what you see?

Scoring guide:
0-1 = Wrong subject entirely, no visual connection to the query
2-3 = Same general category but missing key details, or only tangentially related
4-5 = Partial match: correct subject but wrong context/action, or vice versa
6-7 = Good match: subject and context align, but minor details differ or are unclear
8-9 = Strong match: subject, action, and context all clearly match the query
10 = Perfect match: image directly and unambiguously depicts exactly what the query asks for

Be strict and critical. Do not default to middle scores (4-6) out of uncertainty — commit to a score based on visible evidence only. Ignore information not visible in the image.

Return ONLY a single integer from 0 to 10. No explanation, no text, no punctuation.

Return ONLY a single number from 0 to 10, with no other text.
"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        try:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,          # deterministic, cần cho scoring ổn định
                num_beams=1,               # tránh beam search tốn thêm VRAM
                use_cache=True,
                pad_token_id=self.processor.tokenizer.pad_token_id
                or self.processor.tokenizer.eos_token_id,
            )
        except torch.cuda.OutOfMemoryError:
            print("[VLM] OOM khi generate, giải phóng cache và bỏ qua candidate")
            torch.cuda.empty_cache()
            gc.collect()
            return None

        generated_ids = [
            output_ids[i][inputs["input_ids"].shape[1]:]
            for i in range(len(output_ids))
        ]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        print("[VLM RAW]:", response)

        score = self._parse_score(response)

        # Giải phóng tensor trung gian ngay để tránh tích tụ VRAM
        # khi score() được gọi hàng trăm/nghìn lần liên tiếp (rerank
        # nhiều frame video).
        del inputs, output_ids, generated_ids
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return score

    @staticmethod
    def _parse_score(response: str):
        """Parse số điểm từ output của model một cách bền hơn.

        Model đôi khi trả thêm text/markdown dù đã prompt "ONLY a
        number", nên dùng regex tìm số đầu tiên thay vì float() trực
        tiếp trên toàn bộ string.
        """
        match = re.search(r"-?\d+(\.\d+)?", response)
        if match is None:
            return None
        score = float(match.group())
        return max(0.0, min(10.0, score))
