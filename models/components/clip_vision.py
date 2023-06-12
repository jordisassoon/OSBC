import torch
from transformers import AutoProcessor, CLIPVisionModelWithProjection

class CLIPVisionComponent:
    def __init__(self, model_name) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        return self.model(**inputs).image_embeds