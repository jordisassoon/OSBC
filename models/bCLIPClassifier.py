from .heads.OCR import OCR
from .heads.BERT import BERT
from .heads.CLIP import CLIP
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
nlp = spacy.load("en_core_web_sm")


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
        self.BERT = BERT(model_name="bert-base-cased")
        self.CLIP = CLIP(model_name="RN50")

    def extract_text(self, image):
        return self.OCR.extract_text(image)

    def summarize(self, text):
        return self.BERT.summarize(text)

    def encode_image(self, image):
        return self.CLIP.encode_image(image)

    def encode_description(self, text):
        return self.CLIP.encode_text(text)

    @staticmethod
    def preprocess_text(sentence):
        """
        Lemmatize, lowercase, remove numbers and stop words

        Args:
          sentence: The sentence we want to process.

        Returns:
          A list of processed words
        """
        sentence = [token.lemma_.lower()
                    for token in nlp(sentence)
                    if token.is_alpha]

        return ' '.join(sentence)

    def forward(self, label):
        # returns the image and text embeddings, where the text is extended with OCR
        description = "a photo of the number " + label
        # print(extracted_text)
        # summarized_text = self.summarize(extracted_text)
        # print(summarized_text)
        return Embedding(self.encode_description(description), label)

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

        extracted_text = self.extract_text(image)
        extracted_text = self.preprocess_text(extracted_text)

        if extracted_text == '':
            return np.argmax(clip_similarity)

        model = SentenceTransformer('stsb-mpnet-base-v2')

        sentences = list(map(lambda x: x.get_text(), embeddings))
        sentences.append(extracted_text)

        sentence_embeddings = model.encode(sentences)

        bert_similarity = self.bert_similarity(sentence_embeddings[-1], sentence_embeddings[:-1], 1)

        return np.argmax(clip_similarity + bert_similarity)

    def batch_predict(self, queries, embeddings):
        return list(map(lambda x: self.predict(x, embeddings), queries))
