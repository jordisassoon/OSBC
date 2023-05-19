from models.classifiers.combined_classifier import Comparator
from validation.score import Score
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np

path = "data/mnist/"

df = pd.read_csv(path + "mnist_test.csv")
df = df.dropna()

images = []
truth = []

for i, row in tqdm(df.iterrows()):
    im = Image.fromarray(np.array(row[1:]))
    images.append(im)
    truth.append(row[0])

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
template = "an image of the number: "

comparator = Comparator(clip_model="RN50")
scorer = Score(images, truth)

embeddings = []
for label in labels:
    embeddings.append(comparator.forward(template, label))
comparator.set_embeddings(embeddings)

scores = scorer.score_comparator(comparator)
print("OCR Classifier score: " + str(scores[0]))
print("CLIP Classifier score: " + str(scores[1]))
print("BERT-CLIP Classifier score: " + str(scores[2]))
print("teCLIP Classifier score: " + str(scores[3]))
