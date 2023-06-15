from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor, CLIPTokenizerFast
import torch

class CLIP:
    def __init__(self, pretrained_model) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(pretrained_model).to(self.device)
        self.image_processor = CLIPImageProcessor.from_pretrained(pretrained_model)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(pretrained_model)
        self.processor = CLIPProcessor(self.image_processor, self.tokenizer)
    
    @torch.no_grad()
    def forward(self, images, texts):
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        return logits_per_image.argmax(dim=1)

path = './data/'
print("loading images...")

from dataloaders.classification_loaders.image_loader import ImageLoader

image_loader = ImageLoader(images_dir=path+'characters/validation', image_size=(28, 28))

dataloader = image_loader.get_loader()

texts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
            'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
            'W', 'X', 'Y', 'Z']
template = "an image of the letter: {}"

truth = []
images = []

for i, label in enumerate(texts):
    texts[i] = template.format(label)

for datapoint in image_loader.dataset:
    image, label = datapoint
    images.append(image)
    truth.append(label)

clip_model = CLIP("openai/clip-vit-base-patch32")

import numpy as np
from tqdm import tqdm

preds = np.array([])

for batch in tqdm(dataloader):
    images, labels = batch
    pred = clip_model.forward(images=images, texts=texts).cpu().numpy()
    preds = np.append(preds, pred)

from sklearn.metrics import accuracy_score

print(accuracy_score(truth, preds))