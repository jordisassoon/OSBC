from transformers import CLIPProcessor, CLIPModel
import torch

class CLIP:
    def __init__(self, pretrained_model) -> None:
        from transformers import AutoProcessor, CLIPVisionModelWithProjection
        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(pretrained_model).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(pretrained_model)
        self.text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.text_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.vision_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @torch.no_grad()
    def forward(self, images, texts):
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        return logits_per_image.argmax(dim=1)
    
    @torch.no_grad()
    def embed_images(self, images):
        inputs = self.vision_processor(images=images, return_tensors="pt").to(self.device)

        outputs = self.vision_model(**inputs)

        return outputs.image_embeds
    
    @torch.no_grad()
    def embed_text(self, text):
        inputs = self.text_tokenizer(text, padding=True, return_tensors="pt").to(self.device)

        outputs = self.text_model(**inputs)

        return outputs.text_embeds

from dataloaders.retrieval_loaders.flickr8k_loader import Flickr8kDataset
from torch.utils.data import DataLoader
from torchvision import transforms

path = './data/flickr8k/'
captions = path + "captions.csv"
images = path + "Images"

dataset = Flickr8kDataset(captions, images, 
                             transform=transforms.Compose([
                                 transforms.Resize((256, 256))
                                 ]))

def collate_fn(list_items):
     x = []
     y = []
     for x_, y_ in list_items:
         x.append(x_)
         y.append(y_)
     return x, y

dataloader = DataLoader(dataset=dataset, batch_size=16, num_workers=0, collate_fn=collate_fn)

clip_model = CLIP("openai/clip-vit-base-patch32")

from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

image_embeddings = torch.tensor([]).to(device)

for batch in tqdm(dataloader):
    images, _ = batch
    embeddings = clip_model.embed_images(images=images)
    image_embeddings = torch.cat((image_embeddings, embeddings), 0)

image_embeddings = image_embeddings.T

import numpy as np
predictions = np.array([])

for batch in tqdm(dataloader):
    _, captions = batch
    for caption_list in captions:
        text_embeddings = clip_model.embed_text(text=caption_list)
        predictions = np.append(predictions, torch.matmul(text_embeddings, image_embeddings).argmax(dim=1).cpu().numpy())

labels = dataset.__getlabels__()

from sklearn.metrics import accuracy_score

print("scoring...")

print("OSBC: " + str(accuracy_score(labels, predictions)))