import torch
import open_clip


class SiglipEncoder:
    def __init__(self):
        self.device = "cpu"
        print(f"[*] Đang nạp mô hình SigLIP trên {self.device}...")
        
        # Thay thế ViT-B-32 bằng SigLIP (Sử dụng ViT-B-16-SigLIP hoặc phiên bản mạnh hơn nếu GPU cho phép)
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16-SigLIP", 
            pretrained="webli"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        print("[*] Nạp SigLIP hoàn tất.")

    def encode_text(self, text):
        # SigLIP tokenizer hoạt động tương tự CLIP
        tokens = self.tokenizer([text]).to(self.device)

        with torch.no_grad():
            vector = self.model.encode_text(tokens)

        vector = vector.cpu().numpy().astype("float32")
        return vector[0]