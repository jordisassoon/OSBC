import torch

class OCR:
    def __init__(self, pretrained_model) -> None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = VisionEncoderDecoderModel.from_pretrained(pretrained_model).to(self.device)
        self.processor = TrOCRProcessor.from_pretrained(pretrained_model)

    def forward(self, images) -> list:
        pixel_values  = self.processor(images=images, return_tensors="pt").to(self.device).pixel_values
        generated_ids = self.model.generate(pixel_values)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

path = './data/'
print("loading images...")

from dataloaders.image_loader import ImageLoader

image_loader = ImageLoader(images_dir=path+'characters/validation', image_size=(28, 28))

dataloader = image_loader.get_loader()

ocr_model = OCR(pretrained_model="microsoft/trocr-base-printed")

from tqdm import tqdm

extracted_texts = []

for batch in tqdm(dataloader):
    images, labels = batch
    extracted_texts.extend(ocr_model.forward(images=images))

print(extracted_texts)
