import torch
from transformers import AutoTokenizer, CLIPTextModelWithProjection

class CLIPTextComponent:
    def __init__(self, model_name) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPTextModelWithProjection.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @torch.no_grad()
    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        return self.model(**inputs).text_embeds