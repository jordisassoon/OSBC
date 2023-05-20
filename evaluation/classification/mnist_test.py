from evaluation.classification.classification_test import ClassificationTest
from models.classifiers.combined_classifier import Comparator
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np

path = "../../data/mnist/"

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
tester = ClassificationTest(comparator=comparator, queries=images, truth=truth)
tester.embed(labels=labels, template=template)
tester.test()
