from models.heads.OCR import OCR
from models.heads.BERT import BERT
from models.heads.CLIP import CLIP
import numpy as np
import torch
from tqdm import tqdm


class Embedding:
    def __init__(self, description_embedding):
        self.description = description_embedding

    def get_description(self):
        return self.description


class teCLIPClassifier:
    def __init__(self):
        self.OCR = OCR(config='--psm 10')
        self.BERT = BERT(st_name='all-mpnet-base-v2')
        self.CLIP = CLIP(model_name="RN50")
        self.embeddings = None

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings

    def get_description_tensors(self):
        return torch.stack((list(map(lambda x: x.get_description().squeeze(), self.embeddings))), 0)

    @staticmethod
    def format_and_extend(template, extracted_text, description):
        return description + template.format(extracted_text)

    @torch.no_grad()
    def forward(self, label):
        # returns the image and text embeddings, where the text is extended with OCR
        description = "an image of the letter: " + label
        return Embedding(self.CLIP.encode_text(description))

    @torch.no_grad()
    def predict(self, image, descriptions):
        image_query = self.CLIP.encode_image(image).squeeze()
        image_similarity = self.CLIP.similarity_score(image_query, descriptions)

        extracted_text = self.OCR.extract_text(image)
        extracted_text = self.BERT.preprocess_text(extracted_text, stop_words=True)

        if extracted_text == '':
            return np.argmax(image_similarity)

        encoded_text = self.CLIP.encode_text(extracted_text)

        text_similarity = self.CLIP.similarity_score(encoded_text, descriptions)

        return np.argmax(image_similarity + text_similarity)

    def batch_predict(self, queries):
        descriptions = self.get_description_tensors()
        return list(map(lambda x: self.predict(x, descriptions), tqdm(queries)))
