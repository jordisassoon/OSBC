from models.retrievers.combined_retriever import Comparator
import pandas as pd
from tqdm import tqdm
from PIL import Image
from validation.score import Score

path = "data/dilbert/"

df = pd.read_csv(path + "annotations.csv")
df = df.drop(["col1", "col2", "col3"], axis=1)
df = df.dropna()
filenames = df["original_filename"]
query_texts = df["Comics_text_box"].astype("string")

images = []
truth = []
label = "A comic panel"

for i, filename in tqdm(enumerate(filenames)):
    im = Image.open(path + "panels/" + filename)
    images.append(im)
    truth.append(i)

comparator = Comparator(clip_model="ViT-L/14")
scorer = Score(query_texts, truth)

embeddings = []
for image in tqdm(images):
    embeddings.append(comparator.forward(image, label))
comparator.set_embeddings(embeddings)

scores = scorer.score_comparator(comparator)
print("OCR Classifier score: " + str(scores[0]))
print("CLIP Classifier score: " + str(scores[1]))
print("BERT-CLIP Classifier score: " + str(scores[2]))
print("teCLIP Classifier score: " + str(scores[3]))
