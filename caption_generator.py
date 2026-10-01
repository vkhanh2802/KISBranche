import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering, BlipForConditionalGeneration
from paths import CAPTION_JSON, DATA_DIR, MAPPING_DIR, stored_path

# 1. LOAD CSV MAPPING
def load_frame_mapping(csv_path):
    mapping = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                n = int(row.get("n", row.get("keyframe_n", i)))
                frame_id = int(row.get("frame_idx", row.get("frame_id", 0)))
                mapping[n] = frame_id
            except (ValueError, TypeError):
                continue
    return mapping

# 2. DATASET CHO BATCH PROCESSING
class KeyframeDataset(Dataset):
    def __init__(self, tasks):
        """
        tasks: List of dict chứa {video_id, keyframe_n, frame_id, img_path}
        """
        self.tasks = tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        image = Image.open(task["img_path"]).convert("RGB")
        return image, task

def collate_fn(batch):
    images = [item[0] for item in batch]
    metadata = [item[1] for item in batch]
    return images, metadata

# 3. VQA QUESTIONS
VQA_QUESTIONS = {
    "people": "Are there any people in the image?",
    "clothing_color": "What color clothes are the people wearing?",
    "main_object": "What is the main object in the image?",
    "object_color": "What color is the main object?",
    "action": "What action is happening in the image?",
    "setting": "Is this indoor or outdoor?",
    "location": "Where does this scene take place?",
}

# 4. BATCH GENERATION HELPERS
def generate_batch_captions(images, processor, model, device, max_new_tokens=50):
    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=3, # Giảm beams từ 5 xuống 3 để tăng tốc mà không giảm chất lượng nhiều
            do_sample=False,
            repetition_penalty=1.1,
        )
    return processor.batch_decode(outputs, skip_special_tokens=True)

def generate_batch_vqa(images, question, processor, model, device, max_new_tokens=15):
    # BLIP VQA nhận danh sách ảnh và cùng 1 câu hỏi lặp lại cho toàn bộ batch
    questions = [question] * len(images)
    inputs = processor(images=images, text=questions, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return processor.batch_decode(outputs, skip_special_tokens=True)

# 5. MAIN BATCH GENERATION
def generate_captions_with_mapping(
    image_root=DATA_DIR,
    csv_root=MAPPING_DIR,
    output_json=CAPTION_JSON,
    video_ids=None,
    batch_size=16, # Điều chỉnh batch size tùy VRAM (VD: 8, 16, 32)
    num_workers=4,
):
    image_root = os.fspath(image_root)
    csv_root = os.fspath(csv_root)
    output_json = os.fspath(output_json)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    cap_model_id = "Salesforce/blip-image-captioning-large"
    vqa_model_id = "Salesforce/blip-vqa-base"

    print(f"[*] Loading models...")
    cap_processor = BlipProcessor.from_pretrained(cap_model_id)
    cap_model = BlipForConditionalGeneration.from_pretrained(cap_model_id)

    vqa_processor = BlipProcessor.from_pretrained(vqa_model_id)
    vqa_model = BlipForQuestionAnswering.from_pretrained(vqa_model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cap_model.to(device)
    vqa_model.to(device)

    if device == "cuda":
        cap_model = cap_model.half()
        vqa_model = vqa_model.half()

    cap_model.eval()
    vqa_model.eval()

    print(f"[*] Device: {device} | Batch Size: {batch_size}")

    # Load existing checkpoint
    if os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                caption_data = json.load(f)
        except Exception:
            caption_data = []
    else:
        caption_data = []

    processed = {(x["video_id"], x["frame_id"]) for x in caption_data}
    print(f"[*] Already processed: {len(processed)} frames")

    # Gom toàn bộ công việc cần làm vào 1 danh sách
    valid_extensions = (".jpg", ".jpeg", ".png")
    video_folders = sorted(
        d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))
    )
    if video_ids is not None:
        video_folders = [v for v in video_folders if v in video_ids]

    tasks_to_run = []
    for video_id in video_folders:
        image_dir = os.path.join(image_root, video_id)
        csv_path = os.path.join(csv_root, video_id + ".csv")
        if not os.path.exists(csv_path):
            continue

        frame_mapping = load_frame_mapping(csv_path)
        image_files = sorted(
            f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)
        )

        for img_name in image_files:
            try:
                keyframe_n = int(os.path.splitext(img_name)[0])
            except ValueError:
                continue

            frame_id = frame_mapping.get(keyframe_n)
            if frame_id is None or (video_id, frame_id) in processed:
                continue

            tasks_to_run.append({
                "video_id": video_id,
                "frame_id": frame_id,
                "keyframe_n": keyframe_n,
                "img_path": os.path.join(image_dir, img_name)
            })

    print(f"[*] Total frames remaining to process: {len(tasks_to_run)}")
    if not tasks_to_run:
        print("[*] Nothing to process.")
        return

    # Tạo DataLoader để load ảnh song song
    dataset = KeyframeDataset(tasks_to_run)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=collate_fn
    )

    # Chạy xử lý theo Batch
    save_counter = 0
    for images, metadatas in dataloader:
        try:
            # 1. Sinh Caption tổng quát cho toàn bộ batch
            captions = generate_batch_captions(images, cap_processor, cap_model, device)

            # 2. Hỏi VQA cho toàn bộ batch
            vqa_batch_answers = {i: {} for i in range(len(images))}
            for q_key, question in VQA_QUESTIONS.items():
                answers = generate_batch_vqa(images, question, vqa_processor, vqa_model, device)
                for i, ans in enumerate(answers):
                    vqa_batch_answers[i][q_key] = ans.strip()

            # 3. Đóng gói kết quả cho từng item trong batch
            for i, meta in enumerate(metadatas):
                gen_cap = captions[i].strip()
                answers = vqa_batch_answers[i]

                detail_parts = []
                if answers.get("people") and answers["people"].lower() not in ("no", "none"):
                    detail_parts.append(f"people present, wearing {answers.get('clothing_color', 'unknown')} clothes")
                if answers.get("main_object"):
                    detail_parts.append(f"main object: {answers['main_object']} ({answers.get('object_color', 'unknown')} color)")
                if answers.get("action"):
                    detail_parts.append(f"action: {answers['action']}")
                if answers.get("setting"):
                    detail_parts.append(f"setting: {answers['setting']}")
                if answers.get("location"):
                    detail_parts.append(f"location: {answers['location']}")

                details_text = "; ".join(detail_parts)
                retrieval_text = f"{gen_cap}. {details_text}"

                item = {
                    "video_id": meta["video_id"],
                    "frame_id": meta["frame_id"],
                    "keyframe_n": meta["keyframe_n"],
                    "image_path": stored_path(Path(meta["img_path"])),
                    "caption": gen_cap,
                    "vqa_answers": answers,
                    "details": details_text,
                    "retrieval_text": retrieval_text,
                }
                caption_data.append(item)

            save_counter += 1
            print(f"[*] Processed batch {save_counter}/{-(-len(tasks_to_run)//batch_size)}")

            # Ghi Checkpoint sau mỗi Batch (thay vì sau mỗi ảnh) để tiết kiệm IO
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(caption_data, f, ensure_ascii=False, indent=2)

            if device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"[Error in batch]: {e}")

    print(f"\n[*] Finished! Total saved: {len(caption_data)} captions")


if __name__ == "__main__":
    generate_captions_with_mapping(batch_size=8, num_workers=4)