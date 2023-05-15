from models.heads.OCR import OCR
from models.heads.BERT import BERT
import numpy as np
import torch
from tqdm import tqdm


class BCEmbedding:
    def __init__(self, description_embedding, text):
        self.description = description_embedding
        self.raw_text = text

    def get_description(self):
        return self.description

    def get_text(self):
        return self.raw_text


class BERTRetriever:
    def __init__(self):
        self.OCR = OCR(config=None)
        self.BERT = BERT(st_name='all-mpnet-base-v2')
        self.embeddings = None

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings

    def get_inner_text_tensors(self):
        return self.BERT.encode_text(list(map(lambda x: x.get_text(), self.embeddings)))

    def forward(self, image, description):
        # returns the image and text embeddings, where the text is extended with OCR
        extracted_text = self.OCR.extract_text(image)
        extracted_text = self.BERT.preprocess_text(extracted_text)
        processed_description = self.BERT.preprocess_text(description)
        return BCEmbedding(processed_description, extracted_text)

    @torch.no_grad()
    def predict(self, query, encoded_sentences):
        processed_text = self.BERT.preprocess_text(query)
        encoded_text = self.BERT.encode_text(processed_text)

        bert_ocr_similarity = self.BERT.similarity_score(encoded_text, encoded_sentences, 1)

        return np.argmax(bert_ocr_similarity)

    def batch_predict(self, queries):
        encoded_inner_text = self.get_inner_text_tensors()
        return list(map(lambda x: self.predict(x, encoded_inner_text), tqdm(queries)))
