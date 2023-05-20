from retrieval_test import RetrievalTest
from models.retrievers.combined_retriever import Comparator
import pandas as pd
from tqdm import tqdm
from PIL import Image

path = "../../data/dilbert/"

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
tester = RetrievalTest(comparator=comparator, queries=query_texts, truth=truth)
tester.embed(images=images)
tester.test()

