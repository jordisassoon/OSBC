import torch
import clip
from tqdm import tqdm


class CLIP:
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, self.device)

    def eval(self):
        self.model.to(self.device)
        self.model.eval()

    def encode_image(self, image):
        # preprocesses image and encodes it
        return self.model.encode_image(self.preprocess(image).unsqueeze(0).to(self.device))

    def encode_text(self, text):
        # converts text to tokens and encodes the vector
        return self.model.encode_text(clip.tokenize(text).to(self.device))
