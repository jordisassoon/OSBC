from models.heads.CLIP import CLIP
from tqdm import tqdm
import numpy as np
import torch


class Embedding:
    def __init__(self, description_embedding):
        self.description = description_embedding

    def get_description(self):
        return self.description


class nCLIPClassifier:
    def __init__(self):
        self.CLIP = CLIP(model_name="RN50")
        self.embeddings = None

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings

    def get_description_tensors(self):
        return torch.stack((list(map(lambda x: x.get_description().squeeze(), self.embeddings))), 0)

    @torch.no_grad()
    def forward(self, label):
        # returns the image and text embeddings, where the text is extended with OCR
        description = "an image of the letter: " + label
        return Embedding(self.CLIP.encode_text(description))

    @torch.no_grad()
    def predict(self, image, descriptions):
        clip_query = self.CLIP.encode_image(image).squeeze()

        clip_similarity = self.CLIP.similarity_score(clip_query, descriptions)

        return np.argmax(clip_similarity)

    def batch_predict(self, queries):
        descriptions = self.get_description_tensors()
        return list(map(lambda x: self.predict(x, descriptions), tqdm(queries)))
