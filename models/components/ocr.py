import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import pytesseract
from torchvision.transforms.functional import to_pil_image

class OCRComponent:
    def __init__(self, model_name) -> None:
        if model_name == "psm 10" or model_name == "psm 6":
            self.model_flag = "pytesseract"
            self.config = model_name
            return
        else:
            self.model_flag = "trocr"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.processor = TrOCRProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def forward(self, images) -> list:
        if self.model_flag == "pytesseract":
            return self.pytesseract_forward(images=images)
        pixel_values  = self.processor(images=images, return_tensors="pt", padding=True).to(self.device).pixel_values
        generated_ids = self.model.generate(pixel_values)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    def pytesseract_forward(self, images) -> list:
        custom_config = "--oem 3 --" + self.config
        return [pytesseract.image_to_string(to_pil_image(image), config=custom_config) for image in images]