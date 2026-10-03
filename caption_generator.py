import os
import json
from pathlib import Path
import numpy  # Load the shared Windows OpenMP runtime before PyTorch.
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering, BlipForConditionalGeneration
from artifact_io import atomic_write_json
from caption_schema import (
    load_frame_mapping,
    normalize_caption_record,
    validate_unique_caption_records,
)
from model_config import CAPTION_GENERATION_CONFIG
from paths import (
    CAPTION_DIR,
    DATA_DIR,
    MAPPING_DIR,
    canonical_video_artifact_path,
    discover_video_image_dirs,
    resolve_video_artifact,
    stored_path,
)

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
        with Image.open(task["img_path"]) as source_image:
            image = source_image.convert("RGB")
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
    output_dir=CAPTION_DIR,
    video_ids=None,
    batch_size=16, # Điều chỉnh batch size tùy VRAM (VD: 8, 16, 32)
    num_workers=4,
):
    image_root = os.fspath(image_root)
    csv_root = Path(csv_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Gom toàn bộ công việc cần làm vào 1 danh sách
    valid_extensions = (".jpg", ".jpeg", ".png")
    discovered = discover_video_image_dirs(image_root)
    if video_ids is not None:
        requested = set(video_ids)
        missing = sorted(requested - set(discovered))
        if missing:
            raise FileNotFoundError(
                f"Missing image directories: {', '.join(missing)}"
            )
        discovered = {
            video_id: discovered[video_id]
            for video_id in sorted(requested)
        }
    video_folders = sorted(discovered)

    tasks_to_run = []
    caption_data_by_video = {}
    output_path_by_video = {}
    processed = set()
    for video_id in video_folders:
        image_dir = discovered[video_id]
        csv_path = csv_root / f"{video_id}.csv"
        if not csv_path.is_file():
            continue

        canonical_output = canonical_video_artifact_path(
            output_dir,
            video_id,
            ".json",
        )
        existing_output = resolve_video_artifact(
            output_dir,
            video_id,
            ".json",
        )
        if existing_output.is_file():
            if existing_output.resolve() != canonical_output.resolve():
                raise RuntimeError(
                    f"Legacy caption path detected: {existing_output}. "
                    "Run migrate_caption_artifacts.py --apply first."
                )
            with existing_output.open("r", encoding="utf-8") as file:
                caption_data = json.load(file)
            if not isinstance(caption_data, list):
                raise ValueError(f"Caption checkpoint must be an array: {existing_output}")
        else:
            caption_data = []

        frame_mapping = load_frame_mapping(video_id, csv_root)
        caption_data = [
            normalize_caption_record(record, video_id, frame_mapping)
            for record in caption_data
        ]
        validate_unique_caption_records(caption_data, video_id)
        # Drop placeholder records without usable text (e.g. OCR-created
        # stubs) so their keyframes are regenerated instead of skipped.
        caption_data = [
            item
            for item in caption_data
            if (item.get("retrieval_text") or item.get("caption"))
        ]
        caption_data_by_video[video_id] = caption_data
        output_path_by_video[video_id] = canonical_output
        processed.update(
            (item["video_id"], int(item["keyframe_n"]))
            for item in caption_data
        )

        image_files = sorted(
            f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)
        )

        for img_name in image_files:
            try:
                keyframe_n = int(os.path.splitext(img_name)[0])
            except ValueError:
                continue

            frame_id = frame_mapping.get(keyframe_n)
            if frame_id is None or (video_id, keyframe_n) in processed:
                continue

            tasks_to_run.append({
                "video_id": video_id,
                "frame_id": frame_id,
                "keyframe_n": keyframe_n,
                "img_path": os.path.join(image_dir, img_name)
            })

    print(f"[*] Total frames remaining to process: {len(tasks_to_run)}")
    print(f"[*] Already processed: {len(processed)} frames")
    if not tasks_to_run:
        print("[*] Nothing to process.")
        return

    cap_model_id = CAPTION_GENERATION_CONFIG["caption_model"]
    cap_revision = CAPTION_GENERATION_CONFIG["caption_revision"]
    vqa_model_id = CAPTION_GENERATION_CONFIG["vqa_model"]
    vqa_revision = CAPTION_GENERATION_CONFIG["vqa_revision"]

    print("[*] Loading models...")
    cap_processor = BlipProcessor.from_pretrained(
        cap_model_id,
        revision=cap_revision,
    )
    cap_model = BlipForConditionalGeneration.from_pretrained(
        cap_model_id,
        revision=cap_revision,
    )
    vqa_processor = BlipProcessor.from_pretrained(
        vqa_model_id,
        revision=vqa_revision,
    )
    vqa_model = BlipForQuestionAnswering.from_pretrained(
        vqa_model_id,
        revision=vqa_revision,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        cap_model = cap_model.half().to(device)
        vqa_model = vqa_model.half().to(device)
    else:
        cap_model.to(device)
        vqa_model.to(device)
    cap_model.eval()
    vqa_model.eval()
    print(f"[*] Device: {device} | Batch Size: {batch_size}")

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
            affected_videos = set()
            for i, meta in enumerate(metadatas):
                gen_cap = captions[i].strip()
                answers = vqa_batch_answers[i]

                detail_parts = []
                people_answer = answers.get("people", "").strip().lower()
                if people_answer and not people_answer.startswith("no"):
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
                retrieval_text = f"{gen_cap}. {details_text}".strip(". ")

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
                caption_data_by_video[meta["video_id"]].append(item)
                affected_videos.add(meta["video_id"])

            save_counter += 1
            print(f"[*] Processed batch {save_counter}/{-(-len(tasks_to_run)//batch_size)}")

            for video_id in affected_videos:
                atomic_write_json(
                    output_path_by_video[video_id],
                    caption_data_by_video[video_id],
                )

        except Exception as e:
            raise RuntimeError(f"Caption generation batch failed: {e}") from e
        finally:
            for image in images:
                image.close()

    total_saved = sum(len(items) for items in caption_data_by_video.values())
    print(f"\n[*] Finished! Total saved: {total_saved} captions")


if __name__ == "__main__":
    generate_captions_with_mapping(batch_size=8, num_workers=4)
