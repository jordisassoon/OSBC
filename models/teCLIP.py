from .heads.OCR import OCR
from .heads.BERT import BERT
from .heads.CLIP import CLIP


class teCLIP:
    def __init__(self):
        self.OCR = OCR(config=None)
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
    def format_and_extend(template, extracted_text, description):
        return description + template.format(extracted_text)

    def forward(self, image, description, template):
        # returns the image and text embeddings, where the text is extended with OCR
        extracted_text = self.extract_text(image)
        # print(extracted_text)
        summarized_text = self.summarize(extracted_text)
        # print(summarized_text)
        extended_description = self.format_and_extend(template, summarized_text, description)
        # print(extended_description)
        return self.encode_image(image), self.encode_description(extended_description)
