import torch
from transformers import CLIPProcessor, CLIPModel

class CLIPComponent:
    def __init__(self, model_name) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def forward(self, images, texts):
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)
        return self.model(**inputs).logits_per_image