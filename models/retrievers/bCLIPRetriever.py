from models.heads.OCR import OCR
from models.heads.BERT import BERT
from models.heads.CLIP import CLIP
import numpy as np
import torch
from tqdm import tqdm


class BCEmbedding:
    def __init__(self, image_embedding, description_embedding, text):
        self.image = image_embedding
        self.description = description_embedding
        self.raw_text = text

    def get_image(self):
        return self.image

    def get_description(self):
        return self.description

    def get_text(self):
        return self.raw_text


class bCLIPRetriever:
    def __init__(self, config=None, st_name='all-mpnet-base-v2', clip_model="RN50"):
        self.OCR = OCR(config=config)
        self.BERT = BERT(st_name=st_name)
        self.CLIP = CLIP(model_name=clip_model)
        self.embeddings = None

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings

    def get_image_tensors(self):
        return torch.stack((list(map(lambda x: x.get_image().squeeze(), self.embeddings))), 0)

    def get_inner_text_tensors(self):
        return self.BERT.encode_text(list(map(lambda x: x.get_text(), self.embeddings)))

    @torch.no_grad()
    def forward(self, image, description):
        # returns the image and text embeddings, where the text is extended with OCR
        extracted_text = self.OCR.extract_text(image)
        extracted_text = self.BERT.preprocess_text(extracted_text)
        return BCEmbedding(self.CLIP.encode_image(image),
                           self.CLIP.encode_text(description),
                           extracted_text)

    @torch.no_grad()
    def predict(self, query, images, encoded_sentences):
        clip_query = self.CLIP.encode_text(query)
        clip_image_similarity = self.CLIP.similarity_score(clip_query, images)

        processed_text = self.BERT.preprocess_text(query)
        encoded_text = self.BERT.encode_text(processed_text)
        bert_similarity = self.BERT.similarity_score(encoded_text, encoded_sentences, 1)

        return np.argmax(clip_image_similarity + bert_similarity)

    def batch_predict(self, queries):
        images = self.get_image_tensors()
        sentences = self.get_inner_text_tensors()
        return list(map(lambda x: self.predict(x, images, sentences), tqdm(queries)))
