from models.heads.OCR import OCR
from models.heads.BERT import BERT
import numpy as np
import torch
from tqdm import tqdm


class Embedding:
    def __init__(self, text):
        self.raw_text = text

    def get_text(self):
        return self.raw_text


class OCRClassifier:
    def __init__(self):
        self.OCR = OCR(config='--psm 10')
        self.BERT = BERT(st_name='all-mpnet-base-v2')
        self.embeddings = None

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings

    def get_inner_text_tensors(self):
        return self.BERT.encode_text(list(map(lambda x: x.get_text(), self.embeddings)))

    @torch.no_grad()
    def forward(self, label):
        return Embedding(label)

    @torch.no_grad()
    def predict(self, image, encoded_sentences):
        extracted_text = self.OCR.extract_text(image)
        extracted_text = self.BERT.preprocess_text(extracted_text, stop_words=True)

        encoded_text = self.BERT.encode_text(extracted_text)

        bert_similarity = self.BERT.similarity_score(encoded_text, encoded_sentences, 1)

        return np.argmax(bert_similarity)

    def batch_predict(self, queries):
        inner_texts = self.get_inner_text_tensors()
        return list(map(lambda x: self.predict(x, inner_texts), tqdm(queries)))
