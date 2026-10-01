import os
import json
import time
from PIL import Image
from pathlib import Path
from google import genai
from paths import DATA_DIR, KIS_ROOT, CAPTION_DIR

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
    print("=== BẮT ĐẦU OCR ===")

    valid_exts = (".jpg", ".jpeg", ".png")

    # Lấy tất cả thư mục video
    video_folders = sorted(
        d for d in DATA_DIR.iterdir()
        if d.is_dir()
    )

    print(f"[*] Tìm thấy {len(video_folders)} thư mục video.")

    for image_dir in video_folders:

        vid = image_dir.name
        parent_dir = vid.split("_")[0]

        # Đường dẫn JSON
        json_path = CAPTION_DIR / parent_dir / f"{vid}.json"

        # Nếu JSON nằm trực tiếp trong CAPTION_DIR
        if not json_path.exists():
            fallback = CAPTION_DIR / f"{vid}.json"
            if fallback.exists():
                json_path = fallback

        json_path.parent.mkdir(parents=True, exist_ok=True)

        # Đọc JSON cũ
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    caption_data = json.load(f)
            except Exception:
                caption_data = []
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

        print(f"\n[{vid}] {len(image_files)} ảnh")

        # Xử lý từng ảnh
        for img_name in image_files:

            try:
                keyframe_n = int(
                    os.path.splitext(img_name)[0]
                )
            except ValueError:
                print(f"  [SKIP] Tên ảnh không hợp lệ: {img_name}")
                continue

            img_path = image_dir / img_name

            # Tìm record trong JSON
            item = existing.get(keyframe_n)

            # Nếu chưa có record thì tạo mới
            if item is None:

                try:
                    image_path = str(
                        img_path.relative_to(KIS_ROOT)
                    )
                except ValueError:
                    image_path = str(img_path)

                item = {
                    "video_id": vid,
                    "frame_id": None,
                    "keyframe_n": keyframe_n,
                    "image_path": image_path,
                    "caption": "",
                    "vqa_answers": {},
                    "ocr_text": "",
                    "details": "",
                    "retrieval_text": ""
                }

                caption_data.append(item)
                existing[keyframe_n] = item

            # Resume: đã có OCR thì bỏ qua
            if item.get("ocr_text"):
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
                        f"  ✓ {img_name}: "
                        f"{item['ocr_text']}"
                    )
                else:
                    print(
                        f"  ✓ {img_name}: NO TEXT"
                    )

                # -----------------------------------------
                # Rebuild retrieval_text
                # -----------------------------------------

                caption = item.get("caption", "") or ""
                details = item.get("details", "") or ""
                ocr_text = item.get("ocr_text", "") or ""

                base_text = ". ".join(
                    text.strip(". ")
                    for text in [caption, details]
                    if text.strip()
                )

                if ocr_text:
                    if base_text:
                        item["retrieval_text"] = (
                            f"{base_text}. [OCR]: {ocr_text}"
                        )
                    else:
                        item["retrieval_text"] = (
                            f"[OCR]: {ocr_text}"
                        )
                else:
                    item["retrieval_text"] = base_text

                # -----------------------------------------
                # Lưu JSON ngay sau mỗi ảnh
                # -----------------------------------------

                with open(
                    json_path,
                    "w",
                    encoding="utf-8"
                ) as f:
                    json.dump(
                        caption_data,
                        f,
                        ensure_ascii=False,
                        indent=2
                    )

                time.sleep(4)

            except Exception as e:

                print(
                    f"  ✗ Lỗi [{vid} - {img_name}]: {e}"
                )

                time.sleep(10)

        print(f"[✓] Hoàn tất {vid}")

    print("\n=== HOÀN TẤT OCR ===")


if __name__ == "__main__":
    run_gemini_ocr_all()
