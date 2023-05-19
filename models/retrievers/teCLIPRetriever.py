from models.heads.OCR import OCR
from models.heads.BERT import BERT
from models.heads.CLIP import CLIP
import torch
import numpy as np
from tqdm import tqdm


class CEmbedding:
    def __init__(self, image_embedding, description_embedding):
        self.image = image_embedding
        self.description = description_embedding

    def get_image(self):
        return self.image

    def get_description(self):
        return self.description


class teCLIPRetriever:
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

    @staticmethod
    def format_and_extend(template, extracted_text, description):
        return description + template.format(extracted_text)

    @torch.no_grad()
    def forward(self, image, description, template):
        # returns the image and text embeddings, where the text is extended with OCR
        extracted_text = self.OCR.extract_text(image)
        extracted_text = self.BERT.preprocess_text(extracted_text)
        extended_description = self.format_and_extend(template, extracted_text, description)
        # print(extended_description)
        return CEmbedding(self.CLIP.encode_image(image),
                          self.CLIP.encode_text(extended_description))

    @torch.no_grad()
    def predict(self, query, images):
        clip_query = self.CLIP.encode_text(query)

        clip_image_similarity = self.CLIP.similarity_score(clip_query, images)

        return np.argmax(clip_image_similarity)

    def batch_predict(self, queries):
        images = self.get_image_tensors()
        return list(map(lambda x: self.predict(x, images), tqdm(queries)))
