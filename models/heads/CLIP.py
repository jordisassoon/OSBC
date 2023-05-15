import torch
import clip


class CLIP:
    def __init__(self, model_name):
        self.device = "cpu"
        self.model, self.preprocess = clip.load(model_name, self.device)
        self.eval()

    def eval(self):
        self.model.to(self.device)
        self.model.eval()

    def encode_image(self, image):
        # preprocesses image and encodes it
        return self.model.encode_image(self.preprocess(image).unsqueeze(0).to(self.device))

    def encode_text(self, text):
        # converts text to tokens and encodes the vector
        return self.model.encode_text(clip.tokenize(text).to(self.device))

    @staticmethod
    def similarity_score(query, descriptions, dim=-1):
        cos = torch.nn.CosineSimilarity(dim=dim, eps=1e-6)
        return cos(query, descriptions)
