import torch
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, util
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class SBERT:
    def __init__(self, pretrained_model) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sentence_transformer = SentenceTransformer(pretrained_model)

    @torch.no_grad()
    def encode_text(self, sentences):
        # encodes sentences for similarity scoring
        return self.sentence_transformer.encode(sentences)
    
    @staticmethod
    def similarity_score(queries, sentence_embeddings):
        # computes the similarity between a query and each sentence
        return util.cos_sim(queries, sentence_embeddings)

class OCR:
    def __init__(self, pretrained_model) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = VisionEncoderDecoderModel.from_pretrained(pretrained_model).to(self.device)
        self.processor = TrOCRProcessor.from_pretrained(pretrained_model)

    @torch.no_grad()
    def forward(self, images) -> list:
        pixel_values  = self.processor(images=images, return_tensors="pt", padding=True).to(self.device).pixel_values
        generated_ids = self.model.generate(pixel_values)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

class CLIP:
    def __init__(self, pretrained_model) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(pretrained_model).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(pretrained_model)

    @torch.no_grad()
    def forward(self, images, texts):
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)
        return self.model(**inputs).logits_per_image

path = './data/'
print("loading images...")

from dataloaders.image_loader import ImageLoader

image_loader = ImageLoader(images_dir=path+'characters/train', image_size=(28, 28))

dataloader = image_loader.get_loader()

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
            'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
            'W', 'X', 'Y', 'Z']
template = "an image of the letter: {}"

n_classes = len(labels)

formatted_labels = []

truth = []

for label in labels:
    formatted_labels.append(template.format(label))

for datapoint in image_loader.dataset:
    image, label = datapoint
    truth.append(label)

ocr_model = OCR(pretrained_model="microsoft/trocr-base-printed")
sbert_model = SBERT(pretrained_model='all-mpnet-base-v2')
clip_model = CLIP("clip")

import numpy as np
from tqdm import tqdm

embeds = sbert_model.encode_text(labels)
osbc_preds = np.array([])
ocr_sbert_preds = np.array([])
clip_outs = np.array([])

for batch in tqdm(dataloader):
    images, labels = batch
    
    extracted_texts = ocr_model.forward(images=images)
    batch = sbert_model.encode_text(extracted_texts)

    sbert_preds = sbert_model.similarity_score(batch, embeds)
    clip_preds = clip_model.forward(images=images, texts=formatted_labels).cpu() / 100
    
    m = torch.nn.Threshold(0.3, 0)
    sbert_preds_clipped = m(sbert_preds)
    clip_preds_clipped = m(clip_preds)

    osbc_preds = np.append(osbc_preds, (sbert_preds_clipped + clip_preds_clipped).argmax(dim=1).numpy())
    ocr_sbert_preds = np.append(ocr_sbert_preds, sbert_preds.argmax(dim=1).numpy())
    clip_outs = np.append(clip_outs, clip_preds.argmax(dim=1).numpy())

from sklearn.metrics import accuracy_score

print("OSBC: " + str(accuracy_score(truth, osbc_preds)))
print("OCR-SBERT: " + str(accuracy_score(truth, ocr_sbert_preds)))
print("CLIP: " + str(accuracy_score(truth, clip_outs)))