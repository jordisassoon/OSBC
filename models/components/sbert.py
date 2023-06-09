import torch
import spacy
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util

class SBERTComponent:
    def __init__(self, model_name) -> None:
        self.sentence_transformer = SentenceTransformer(model_name)
        self.nlp = spacy.load("en_core_web_sm")

    def process_text(self, sentence, numeric=False, stop_words=False):
        # lemmatize, lowercase, remove numbers and stop words
        if not numeric:
            sentence = [token.lemma_.lower() for token in self.nlp(sentence) if token.is_alpha]
        else:
            sentence = [token.lemma_.lower() for token in self.nlp(sentence)]
        if not stop_words:
            sentence = [token for token in sentence if token not in stopwords.words("english")]
        return ' '.join(sentence)

    @torch.no_grad()
    def encode_text(self, sentences):
        # encodes sentences for similarity scoring
        return self.sentence_transformer.encode(sentences)
    
    @staticmethod
    def similarity_score(queries, sentence_embeddings):
        # computes the similarity between a query and each sentence
        return (util.cos_sim(queries, sentence_embeddings) + 1) / 2