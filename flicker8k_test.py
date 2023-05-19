from models.retrievers.combined_retriever import Comparator
import pandas as pd
from tqdm import tqdm
from PIL import Image
from validation.score import Score

path = "data/flicker8k/"

df = pd.read_csv(path + "captions.csv")
df = df.dropna()
filenames = df["image"]
query_texts = df["caption"].astype("string")

images = []
truth = []

last_filename = None
last_index = -1
for filename in tqdm(filenames):
    if filename == last_filename:
        truth.append(last_index)
    else:
        im = Image.open(path + "Images/" + filename)
        images.append(im)
        last_index += 1
        truth.append(last_index)
        last_filename = filename

comparator = Comparator()
scorer = Score(query_texts, truth)

embeddings = []
for image in tqdm(images):
    embeddings.append(comparator.forward(image))
comparator.set_embeddings(embeddings)

scores = scorer.score_comparator(comparator)
print("OCR Classifier score: " + str(scores[0]))
print("CLIP Classifier score: " + str(scores[1]))
print("BERT-CLIP Classifier score: " + str(scores[2]))
print("teCLIP Classifier score: " + str(scores[3]))
