import numpy  # Load the shared Windows OpenMP runtime before PyTorch.
import torch
import open_clip
from model_config import VISUAL_CONFIG


class SiglipEncoder:
    def __init__(self):
        self.device = "cpu"
        print(f"[*] Loading SigLIP on {self.device}...")
        
        # Thay thế ViT-B-32 bằng SigLIP (Sử dụng ViT-B-16-SigLIP hoặc phiên bản mạnh hơn nếu GPU cho phép)
        self.model, _, _ = open_clip.create_model_and_transforms(
            VISUAL_CONFIG["model"],
            pretrained=VISUAL_CONFIG["pretrained"],
        )
        self.tokenizer = open_clip.get_tokenizer(VISUAL_CONFIG["model"])
        
        self.model = self.model.to(self.device)
        self.model.eval()
        print("[*] SigLIP loaded.")

    def encode_text(self, text):
        # SigLIP tokenizer hoạt động tương tự CLIP
        tokens = self.tokenizer([text]).to(self.device)

        with torch.no_grad():
            vector = self.model.encode_text(tokens)

        vector = vector.cpu().numpy().astype("float32")
        return vector[0]
