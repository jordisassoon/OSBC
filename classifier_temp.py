from transformers import CLIPProcessor, CLIPModel
import torch

model = CLIPModel.from_pretrained("clip")
processor = CLIPProcessor.from_pretrained("clip")

device = "cuda" if torch.cuda.is_available() else "cpu"

path = './data/'
print("loading images...")

from dataloaders.image_loader import ImageLoader

image_loader = ImageLoader(images_dir=path+'characters/validation')

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
            'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
            'W', 'X', 'Y', 'Z']
template = "an image of the letter: {}"

truth = []
images = []

for i, label in enumerate(labels):
    labels[i] = template.format(label)

for datapoint in image_loader.dataset:
    image, label = datapoint
    images.append(image)
    truth.append(label)

model = model.to(device)

print("running inference")

def batch_predict(images, labels, processor, device, model):
    inputs = processor(text=labels, images=images, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    return logits_per_image.argmax(dim=1)

import numpy as np

preds = np.array([])

for i in range(26):
    start = 28 * i
    end = 28 * (i + 1)
    pred = batch_predict(images=images[start:end], labels=labels, processor=processor, device=device, model=model).cpu().numpy()
    preds = np.append(preds, pred)

from sklearn.metrics import accuracy_score

print(accuracy_score(truth, preds))
