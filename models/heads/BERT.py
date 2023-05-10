from transformers import BertTokenizer
from summarizer import Summarizer


class BERT:
    def __init__(self, model_name):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # self.summarizer = Summarizer()

    def encode_text(self, text):
        # converts text to tokens and encodes the vector
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

    def summarize(self, text):
        # summary = self.summarizer(text, num_sentences=1)
        return ''.join(text)
