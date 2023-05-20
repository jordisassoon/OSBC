from retrieval_test import RetrievalTest
from models.retrievers.combined_retriever import Comparator
import pandas as pd
from tqdm import tqdm
from PIL import Image


path = "../../data/flicker8k/"

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
tester = RetrievalTest(comparator=comparator, queries=query_texts, truth=truth)
tester.embed(images=images)
tester.test()
