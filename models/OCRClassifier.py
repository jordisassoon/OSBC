from .heads.OCR import OCR
from .heads.BERT import BERT
from .heads.CLIP import CLIP
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


class OCRClassifier:
    def __init__(self):
        self.OCR = OCR(config='--psm 10')
        self.BERT = BERT(tokenizer_name="bert-base-cased", st_name='all-mpnet-base-v2')
        self.CLIP = CLIP(model_name="ViT-L/14")
        self.embeddings = None

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings

    def get_description_tensors(self):
        return torch.stack((list(map(lambda x: x.get_description().squeeze(), self.embeddings))), 0)

    def get_inner_text_tensors(self):
        return self.BERT.encode_sentences(list(map(lambda x: x.get_text(), self.embeddings)))

    @torch.no_grad()
    def forward(self, label):
        # returns the image and text embeddings, where the text is extended with OCR
        description = "an image of the letter: " + label
        return Embedding(self.CLIP.encode_text(description), label)

    @torch.no_grad()
    def predict(self, image, encoded_sentences):
        extracted_text = self.OCR.extract_text(image)
        extracted_text = self.BERT.preprocess_text(extracted_text, stop_words=True)

        encoded_text = self.BERT.encode_sentences(extracted_text)

        bert_similarity = self.BERT.similarity_score(encoded_text, encoded_sentences, 1)

        return np.argmax(bert_similarity)

    def batch_predict(self, queries):
        inner_texts = self.get_inner_text_tensors()
        return list(map(lambda x: self.predict(x, inner_texts), tqdm(queries)))
