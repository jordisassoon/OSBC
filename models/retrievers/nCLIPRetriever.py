from models.heads.CLIP import CLIP
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


class nCLIPRetriever:
    def __init__(self):
        self.CLIP = CLIP(model_name="RN50")
        self.embeddings = None

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings

    def forward(self, image, description):
        return CEmbedding(self.CLIP.encode_image(image), self.CLIP.encode_text(description))

    def get_image_tensors(self):
        return torch.stack((list(map(lambda x: x.get_image().squeeze(), self.embeddings))), 0)

    def get_description_tensors(self):
        return torch.stack((list(map(lambda x: x.get_description().squeeze(), self.embeddings))), 0)

    @torch.no_grad()
    def predict(self, query, descriptions, images):
        clip_query = self.CLIP.encode_text(query)

        clip_text_similarity = self.CLIP.similarity_score(clip_query, descriptions)
        clip_image_similarity = self.CLIP.similarity_score(clip_query, images)

        return np.argmax(clip_text_similarity + clip_image_similarity)

    def batch_predict(self, queries):
        descriptions = self.get_description_tensors()
        images = self.get_image_tensors()
        return list(map(lambda x: self.predict(x, descriptions, images), queries))
