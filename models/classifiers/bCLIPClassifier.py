from models.heads.OCR import OCR
from models.heads.BERT import BERT
from models.heads.CLIP import CLIP
import numpy as np
import torch
from tqdm import tqdm


class Embedding:
    def __init__(self, description_embedding, text):
        self.description = description_embedding
        self.raw_text = text

    def get_description(self):
        return self.description

    def get_text(self):
        return self.raw_text


class bCLIPClassifier:
    def __init__(self):
        self.OCR = OCR(config='--psm 10')
        self.BERT = BERT(st_name='all-mpnet-base-v2')
        self.CLIP = CLIP(model_name="RN50")
        self.embeddings = None

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings

    def get_description_tensors(self):
        return torch.stack((list(map(lambda x: x.get_description().squeeze(), self.embeddings))), 0)

    def get_inner_text_tensors(self):
        return self.BERT.encode_text(list(map(lambda x: x.get_text(), self.embeddings)))

    @torch.no_grad()
    def forward(self, label):
        # returns the image and text embeddings, where the text is extended with OCR
        description = "an image of the letter: " + label
        return Embedding(self.CLIP.encode_text(description), label)

    @torch.no_grad()
    def predict(self, image, descriptions, encoded_sentences):
        clip_query = self.CLIP.encode_image(image).squeeze()
        clip_similarity = self.CLIP.similarity_score(clip_query, descriptions)

        extracted_text = self.OCR.extract_text(image)
        extracted_text = self.BERT.preprocess_text(extracted_text, stop_words=True)

        if extracted_text == '':
            return np.argmax(clip_similarity)

        encoded_text = self.BERT.encode_text(extracted_text)

        bert_similarity = self.BERT.similarity_score(encoded_text, encoded_sentences, 1)

        return np.argmax(clip_similarity + bert_similarity)

    def batch_predict(self, queries):
        descriptions = self.get_description_tensors()
        inner_texts = self.get_inner_text_tensors()
        return list(map(lambda x: self.predict(x, descriptions, inner_texts), tqdm(queries)))
