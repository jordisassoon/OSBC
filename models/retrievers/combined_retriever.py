from models.heads.OCR import OCR
from models.heads.BERT import BERT
from models.heads.CLIP import CLIP
import numpy as np
import torch
from tqdm import tqdm


class Comparator:
    def __init__(self, config=None, st_name='all-mpnet-base-v2', clip_model="RN50"):
        self.OCR = OCR(config=config)
        self.BERT = BERT(st_name=st_name)
        self.CLIP = CLIP(model_name=clip_model)
        self.clip_images = []
        self.clip_descriptions = []
        self.clip_extended_descriptions = []
        self.bert_inner_text = []
        self.bert_descriptions = []
        self.embeddings = None

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings

    def get_image_tensors(self):
        return torch.stack(self.clip_images, 0)

    def get_inner_text_tensors(self):
        return self.BERT.encode_text(self.bert_inner_text)

    @staticmethod
    def format_and_extend(template, extracted_text, description):
        return description + template.format(extracted_text)

    @torch.no_grad()
    def forward(self, image, description=None, template=None):
        # returns the image and text embeddings, where the text is extended with OCR
        extracted_text = self.OCR.extract_text(image)
        extracted_text = self.BERT.preprocess_text(extracted_text)
        # extended_description = self.format_and_extend(template, extracted_text, description)
        self.clip_images.append(self.CLIP.encode_image(image).squeeze())
        # self.clip_descriptions.append(self.CLIP.encode_text(description))
        # self.clip_extended_descriptions.append(self.CLIP.encode_text(extended_description))
        self.bert_inner_text.append(extracted_text)
        # self.bert_descriptions.append(self.BERT.preprocess_text(description))

    @torch.no_grad()
    def predict(self, query, encoded_images, inner_texts):
        clip_query = self.CLIP.encode_text(query)
        clip_image_similarity = self.CLIP.similarity_score(clip_query, encoded_images)

        nCLIP_prediction = teCLIP_prediction = np.argmax(clip_image_similarity)

        processed_text = self.BERT.preprocess_text(query)
        encoded_text = self.BERT.encode_text(processed_text)
        bert_similarity = self.BERT.similarity_score(encoded_text, inner_texts, 1)

        bert_prediction = np.argmax(bert_similarity)

        bCLIP_prediction = np.argmax(clip_image_similarity + bert_similarity)

        return [bert_prediction, nCLIP_prediction, bCLIP_prediction, teCLIP_prediction]

    def batch_predict(self, queries):
        images = self.get_image_tensors()
        sentences = self.get_inner_text_tensors()
        return list(map(lambda x: self.predict(x, images, sentences), tqdm(queries)))
