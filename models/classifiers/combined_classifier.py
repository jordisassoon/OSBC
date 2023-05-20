from models.heads.OCR import OCR
from models.heads.BERT import BERT
from models.heads.CLIP import CLIP
from models.embeddings.embedding import Embedding
from tqdm import tqdm
import numpy as np
import torch


class Comparator:
    def __init__(self, config='--psm 10', st_name='all-mpnet-base-v2', clip_model="RN50"):
        self.OCR = OCR(config=config)
        self.BERT = BERT(st_name=st_name)
        self.CLIP = CLIP(model_name=clip_model)
        self.CLIP.eval()
        self.embeddings = None

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings

    def get_description_tensors(self):
        return torch.stack((list(map(lambda x: x.get_description().squeeze(), self.embeddings))), 0)

    def get_inner_text_tensors(self):
        return self.BERT.encode_text(list(map(lambda x: x.get_inner_text(), self.embeddings)))

    @torch.no_grad()
    def forward(self, template, label):
        # returns the image and text embeddings, where the text is extended with OCR
        description = template + label
        encoded_description = self.CLIP.encode_text(description)
        processed_label = self.BERT.preprocess_text(label, stop_words=True)
        return Embedding(description_embedding=encoded_description, inner_text=processed_label)

    @torch.no_grad()
    def predict(self, image, descriptions, inner_texts):
        clip_query = self.CLIP.encode_image(image).squeeze()
        clip_similarity = self.CLIP.similarity_score(clip_query, descriptions)

        nCLIP_prediction = np.argmax(clip_similarity)

        extracted_text = self.OCR.extract_text(image)
        extracted_text = self.BERT.preprocess_text(extracted_text, stop_words=True)
        encoded_text = self.BERT.encode_text(extracted_text)
        ocr_bert_similarity = self.BERT.similarity_score(encoded_text, inner_texts, 1)

        bert_prediction = np.argmax(ocr_bert_similarity)

        if extracted_text == '':
            return [bert_prediction, nCLIP_prediction, nCLIP_prediction, nCLIP_prediction]

        teclip_encoded_text = self.CLIP.encode_text(extracted_text)
        teclip_text_similarity = self.CLIP.similarity_score(teclip_encoded_text, descriptions)

        bCLIP_prediction = np.argmax(clip_similarity + ocr_bert_similarity)
        teCLIP_prediction = np.argmax(clip_similarity + teclip_text_similarity)

        return [bert_prediction, nCLIP_prediction, bCLIP_prediction, teCLIP_prediction]

    def batch_predict(self, queries):
        descriptions = self.get_description_tensors()
        inner_texts = self.get_inner_text_tensors()
        return list(map(lambda x: self.predict(x, descriptions, inner_texts), tqdm(queries)))
