import os
import json
import time
from PIL import Image
from google import genai
from artifact_io import atomic_write_json
from caption_schema import (
    load_frame_mapping,
    normalize_caption_record,
    rebuild_retrieval_text,
)
from paths import (
    CAPTION_DIR,
    DATA_DIR,
    resolve_video_artifact,
    stored_path,
    video_caption_path,
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_ID = "gemini-3.1-flash-lite"


def clean_ocr(text):
    """Chuẩn hóa OCR thành một dòng, ngăn cách bằng dấu phẩy."""
    if not text:
        return ""

    return ", ".join(
        line.strip()
        for line in text.splitlines()
        if line.strip()
    )


def run_gemini_ocr_all():
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Missing GEMINI_API_KEY environment variable."
        )

    client = genai.Client(api_key=GEMINI_API_KEY)
    print("=== STARTING OCR ===")

    valid_exts = (".jpg", ".jpeg", ".png")
    failures = []

    # Lấy tất cả thư mục video
    video_folders = sorted(
        d for d in DATA_DIR.iterdir()
        if d.is_dir()
    )

    print(f"[*] Found {len(video_folders)} video directories.")

    for image_dir in video_folders:

        vid = image_dir.name
        json_path = resolve_video_artifact(CAPTION_DIR, vid, ".json")
        canonical_path = video_caption_path(vid)
        if json_path.is_file() and json_path.resolve() != canonical_path.resolve():
            raise RuntimeError(
                f"Legacy caption path detected: {json_path}. "
                "Run migrate_caption_artifacts.py --apply first."
            )
        json_path = canonical_path
        frame_mapping = load_frame_mapping(vid)

        # Đọc JSON cũ
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                loaded_caption_data = json.load(f)
            if not isinstance(loaded_caption_data, list):
                raise ValueError(f"Caption file must be an array: {json_path}")
            caption_data = [
                normalize_caption_record(item, vid, frame_mapping)
                for item in loaded_caption_data
            ]
            if caption_data != loaded_caption_data:
                atomic_write_json(json_path, caption_data)
        else:
            caption_data = []

        # Map keyframe_n -> record
        existing = {
            int(item["keyframe_n"]): item
            for item in caption_data
            if item.get("keyframe_n") is not None
        }

        # Lấy danh sách ảnh
        image_files = sorted(
            f for f in os.listdir(image_dir)
            if f.lower().endswith(valid_exts)
        )

        if not image_files:
            continue

        print(f"\n[{vid}] {len(image_files)} images")

        # Xử lý từng ảnh
        for img_name in image_files:

            try:
                keyframe_n = int(
                    os.path.splitext(img_name)[0]
                )
            except ValueError:
                print(f"  [SKIP] Invalid image filename: {img_name}")
                continue

            img_path = image_dir / img_name

            # Tìm record trong JSON
            item = existing.get(keyframe_n)

            # Nếu chưa có record thì tạo mới
            if item is None:

                frame_id = frame_mapping.get(keyframe_n)
                if frame_id is None:
                    raise ValueError(
                        f"Missing frame mapping for {vid} keyframe {keyframe_n}"
                    )

                item = {
                    "video_id": vid,
                    "frame_id": frame_id,
                    "keyframe_n": keyframe_n,
                    "image_path": stored_path(img_path),
                    "caption": "",
                    "vqa_answers": {},
                    "details": "",
                    "retrieval_text": ""
                }

                caption_data.append(item)
                existing[keyframe_n] = item

            # Field presence distinguishes NO_TEXT from unprocessed records.
            if "ocr_text" in item:
                continue

            try:
                prompt = (
                    "Extract all readable text in this image. "
                    "Only return the extracted text. "
                    "If there is no text, return exactly 'NO_TEXT'."
                )

                # Gửi ảnh cho Gemini
                with Image.open(img_path) as img:
                    response = client.models.generate_content(
                        model=MODEL_ID,
                        contents=[img, prompt]
                    )

                # Lấy kết quả OCR
                ocr = response.text.strip() if response.text else ""

                # Chuẩn hóa OCR:
                # "A\nB\nC" -> "A, B, C"
                ocr = clean_ocr(ocr)

                # Lưu OCR trực tiếp vào field riêng
                if ocr.upper() == "NO_TEXT":
                    item["ocr_text"] = ""
                else:
                    item["ocr_text"] = ocr

                # Hiển thị kết quả
                if item["ocr_text"]:
                    print(
                        f"  [OK] {img_name}: "
                        f"{item['ocr_text']}"
                    )
                else:
                    print(
                        f"  [OK] {img_name}: NO TEXT"
                    )

                item["retrieval_text"] = rebuild_retrieval_text(item)

                # -----------------------------------------
                # Lưu JSON ngay sau mỗi ảnh
                # -----------------------------------------

                atomic_write_json(json_path, caption_data)

                time.sleep(4)

            except Exception as e:
                failures.append((vid, img_name, str(e)))

                print(
                    f"  [ERROR] [{vid} - {img_name}]: {e}"
                )

                time.sleep(10)

        print(f"[OK] Completed {vid}")

    print("\n=== OCR COMPLETE ===")
    if failures:
        raise RuntimeError(f"OCR failed for {len(failures)} images")


if __name__ == "__main__":
    run_gemini_ocr_all()
