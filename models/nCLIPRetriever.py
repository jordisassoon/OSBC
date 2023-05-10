from .heads.CLIP import CLIP
import numpy as np
import torch


class CEmbedding:
    def __init__(self, image_embedding, description_embedding):
        self.image = image_embedding
        self.description = description_embedding

    def get_image(self):
        return self.image

    def get_description(self):
        return self.description


class nCLIP:
    def __init__(self):
        self.CLIP = CLIP(model_name="RN50")

    def encode_image(self, image):
        return self.CLIP.encode_image(image)

    def encode_description(self, text):
        return self.CLIP.encode_text(text)

    def forward(self, image, description):
        return CEmbedding(self.encode_image(image), self.encode_description(description))

    @staticmethod
    def clip_similarity(query, descriptions, dim=1):
        cos = torch.nn.CosineSimilarity(dim=dim, eps=1e-6)
        return list(map(lambda x: cos(query, x).item(), descriptions))

    @torch.no_grad()
    def predict(self, query, embeddings):

        clip_query = self.encode_description(query)

        descriptions = list(map(lambda x: x.get_description(), embeddings))

        clip_similarity = self.clip_similarity(clip_query, descriptions)

        return np.argmax(clip_similarity)

    def batch_predict(self, queries, embeddings):
        return list(map(lambda x: self.predict(x, embeddings), queries))
