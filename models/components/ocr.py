import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class OCRComponent:
    def __init__(self, model_name) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.processor = TrOCRProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def forward(self, images) -> list:
        pixel_values  = self.processor(images=images, return_tensors="pt", padding=True).to(self.device).pixel_values
        generated_ids = self.model.generate(pixel_values)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)