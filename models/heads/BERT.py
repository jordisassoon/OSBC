from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
import spacy


class BERT:
    def __init__(self, tokenizer_name, st_name):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.sentence_transformer = SentenceTransformer(st_name)
        self.nlp = spacy.load("en_core_web_sm")

    def preprocess_text(self, sentence, stop_words=False):
        # lemmatize, lowercase, remove numbers and stop words
        sentence = [token.lemma_.lower() for token in self.nlp(sentence) if token.is_alpha]
        if not stop_words:
            sentence = [token for token in sentence if token not in stopwords.words("english")]
        return ' '.join(sentence)

    def encode_text(self, text):
        # converts text to tokens and encodes the vector
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

    def encode_sentences(self, sentences):
        # encodes sentences for similarity scoring
        return self.sentence_transformer.encode(sentences)

    @staticmethod
    def similarity_score(query, sentence_embeddings, alpha):
        # computes the similarity between a query and each sentence
        return util.cos_sim([query], sentence_embeddings)[0] * alpha
