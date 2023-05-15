from models.retrievers.nCLIPRetriever import nCLIPRetriever
from models.retrievers.bCLIPRetriever import bCLIPRetriever
from models.retrievers.teCLIPRetriever import teCLIPRetriever
from models.retrievers.BERTRetriever import BERTRetriever
import pandas as pd
from tqdm import tqdm
from PIL import Image
from validation.score import Score
import torch

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

embeddings_bert = []
embeddings_bclip = []
embeddings_nclip = []
embeddings_teclip = []
bert = BERTRetriever()  # acc = 0.7346570397111913
bCLIP = bCLIPRetriever()
nCLIP = nCLIPRetriever()
teCLIP = teCLIPRetriever()

template = ", containing the text: \"{}\""

for image in tqdm(images):
    embeddings_bert.append(bert.forward(image, label))
    embeddings_bclip.append(bCLIP.forward(image, label))
    embeddings_nclip.append(nCLIP.forward(image, label))
    embeddings_teclip.append(teCLIP.forward(image, label, template))

bert.set_embeddings(embeddings_bert)
bCLIP.set_embeddings(embeddings_bclip)
nCLIP.set_embeddings(embeddings_nclip)
teCLIP.set_embeddings(embeddings_teclip)

scorer = Score(query_texts, truth)

oscore = scorer.score(bert)
nscore = scorer.score(nCLIP)
bscore = scorer.score(bCLIP)
tescore = scorer.score(teCLIP)

print("OCR Classifier score: " + str(oscore))
print("CLIP Classifier score: " + str(nscore))
print("BERT-CLIP Classifier score: " + str(bscore))
print("teCLIP Classifier score: " + str(tescore))
