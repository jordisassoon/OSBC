from .heads.CLIP import CLIP
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
nlp = spacy.load("en_core_web_sm")


class Embedding:
    def __init__(self, description_embedding):
        self.description = description_embedding

    def get_description(self):
        return self.description


class nCLIPClassifier:
    def __init__(self):
        self.CLIP = CLIP(model_name="RN50")

    def encode_image(self, image):
        return self.CLIP.encode_image(image)

    def encode_description(self, text):
        return self.CLIP.encode_text(text)

    def forward(self, label):
        # returns the image and text embeddings, where the text is extended with OCR
        description = "a photo of the number " + label
        # print(extracted_text)
        # summarized_text = self.summarize(extracted_text)
        # print(summarized_text)
        return Embedding(self.encode_description(description))

    @staticmethod
    def clip_similarity(query, descriptions, dim=1):
        cos = torch.nn.CosineSimilarity(dim=dim, eps=1e-6)
        return list(map(lambda x: cos(query, x).item(), descriptions))

    @staticmethod
    def bert_similarity(query, sentence_embeddings, alpha):
        return cosine_similarity([query], sentence_embeddings)[0] * alpha

    @torch.no_grad()
    def predict(self, image, embeddings):
        clip_query = self.encode_image(image)

        descriptions = list(map(lambda x: x.get_description(), embeddings))

        clip_similarity = self.clip_similarity(clip_query, descriptions)

        return np.argmax(clip_similarity)

    def batch_predict(self, queries, embeddings):
        return list(map(lambda x: self.predict(x, embeddings), queries))
