import torch
from sentence_transformers import SentenceTransformer, util

class SBERTComponent:
    def __init__(self, model_name) -> None:
        self.sentence_transformer = SentenceTransformer(model_name)

    @torch.no_grad()
    def encode_text(self, sentences):
        # encodes sentences for similarity scoring
        return self.sentence_transformer.encode(sentences)
    
    @staticmethod
    def similarity_score(queries, sentence_embeddings):
        # computes the similarity between a query and each sentence
        return util.cos_sim(queries, sentence_embeddings)