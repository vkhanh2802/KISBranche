import numpy  # Load the shared Windows OpenMP runtime before PyTorch.
import torch
import open_clip
from huggingface_hub import snapshot_download
from open_clip.tokenizer import HFTokenizer
from model_config import VISUAL_CONFIG


class SiglipEncoder:
    def __init__(self):
        self.device = "cpu"
        print(f"[*] Loading SigLIP on {self.device}...")
        snapshot_path = snapshot_download(
            repo_id=VISUAL_CONFIG["hf_repo"],
            revision=VISUAL_CONFIG["hf_revision"],
        )
        weights_path = snapshot_path + "/open_clip_model.safetensors"
        self.model, _, self.image_preprocess = open_clip.create_model_and_transforms(
            VISUAL_CONFIG["model"],
            pretrained=weights_path,
        )
        self.tokenizer = HFTokenizer(
            snapshot_path,
            context_length=int(self.model.context_length),
            clean="canonicalize",
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        print("[*] SigLIP loaded.")

    def encode_text(self, text):
        tokens = self.tokenizer([text]).to(self.device)

        with torch.no_grad():
            vector = self.model.encode_text(tokens)

        vector = vector.cpu().numpy().astype("float32")
        return vector[0]

    def encode_images(self, images):
        """Encode RGB PIL images with the same pinned SigLIP as text search."""
        if not images:
            return numpy.empty((0, int(VISUAL_CONFIG["dimension"])), dtype="float32")
        inputs = torch.stack([self.image_preprocess(image) for image in images])
        with torch.inference_mode():
            vectors = self.model.encode_image(inputs.to(self.device))
            vectors = torch.nn.functional.normalize(vectors.float(), dim=-1)
        return vectors.cpu().numpy().astype("float32")
