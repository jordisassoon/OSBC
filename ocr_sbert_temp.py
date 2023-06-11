import torch
from sentence_transformers import SentenceTransformer, util

class SBERT:
    def __init__(self, pretrained_model) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sentence_transformer = SentenceTransformer(pretrained_model)

    def encode_text(self, sentences):
        # encodes sentences for similarity scoring
        return self.sentence_transformer.encode(sentences)
    
    @staticmethod
    def similarity_score(queries, sentence_embeddings):
        # computes the similarity between a query and each sentence
        return util.cos_sim(queries, sentence_embeddings)

class OCR:
    def __init__(self, pretrained_model) -> None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = VisionEncoderDecoderModel.from_pretrained(pretrained_model).to(self.device)
        self.processor = TrOCRProcessor.from_pretrained(pretrained_model)

    def forward(self, images) -> list:
        pixel_values  = self.processor(images=images, return_tensors="pt", padding=True).to(self.device).pixel_values
        generated_ids = self.model.generate(pixel_values)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

path = './data/'
print("loading images...")

from dataloaders.image_loader import ImageLoader

image_loader = ImageLoader(images_dir=path+'characters/validation', image_size=(28, 28))

dataloader = image_loader.get_loader()

ocr_model = OCR(pretrained_model="microsoft/trocr-base-printed")

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
            'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
            'W', 'X', 'Y', 'Z']

sbert_model = SBERT(pretrained_model='all-mpnet-base-v2')

embeds = sbert_model.encode_text(labels)

from tqdm import tqdm
import numpy as np

preds = np.array([])

for batch in tqdm(dataloader):
    images, labels = batch
    extracted_texts = ocr_model.forward(images=images)
    batch = sbert_model.encode_text(extracted_texts)
    preds = np.append(preds, sbert_model.similarity_score(batch, embeds).argmax(dim=1).numpy())

from sklearn.metrics import accuracy_score

truth = []

for datapoint in image_loader.dataset:
    image, label = datapoint
    truth.append(label)

print(accuracy_score(truth, preds))