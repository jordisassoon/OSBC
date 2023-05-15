from models.retrievers.nCLIPRetriever import nCLIPRetriever
from models.retrievers.bCLIPRetriever import bCLIPRetriever
from models.retrievers.teCLIPRetriever import teCLIPRetriever
from models.retrievers.BERTRetriever import BERTRetriever
from sklearn.metrics import accuracy_score
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

embeddings_bert = []
embeddings_bclip = []
embeddings_nclip = []
embeddings_teclip = []
bert = BERTRetriever()
bCLIP = bCLIPRetriever()
nCLIP = nCLIPRetriever()
teCLIP = teCLIPRetriever()

template = ", containing the text: \"{}\""

text = "a comic of Dilbert"
image = Image.open("Demo.jpg")

embeddings_bert.append(bert.forward(image=image, description=text))
embeddings_bclip.append(bCLIP.forward(image=image, description=text))
embeddings_nclip.append(nCLIP.forward(image=image, description=text))
embeddings_teclip.append(teCLIP.forward(image=image, description=text, template=template))

text = "a comic of Dilbert and his friend"
image = Image.open("Demo2.jpg")

embeddings_bert.append(bert.forward(image=image, description=text))
embeddings_bclip.append(bCLIP.forward(image=image, description=text))
embeddings_nclip.append(nCLIP.forward(image=image, description=text))
embeddings_teclip.append(teCLIP.forward(image=image, description=text, template=template))

text = "a diagram"
image = Image.open("CLIP.png")

embeddings_bert.append(bert.forward(image=image, description=text))
embeddings_bclip.append(bCLIP.forward(image=image, description=text))
embeddings_nclip.append(nCLIP.forward(image=image, description=text))
embeddings_teclip.append(teCLIP.forward(image=image, description=text, template=template))

queries = ["A comic of Dilbert talking about passwords",
           "A comic of Dilbert talking about blockchain",
           "A diagram of CLIP"]

labels = [0, 1, 2]

bert.set_embeddings(embeddings_bert)
bCLIP.set_embeddings(embeddings_bclip)
nCLIP.set_embeddings(embeddings_nclip)
teCLIP.set_embeddings(embeddings_teclip)

print("")
print("==== OCR BERT Accuracy ====")
print(accuracy_score(labels, bert.batch_predict(queries)))
print("")
print("==== CLIP Accuracy ====")
print(accuracy_score(labels, nCLIP.batch_predict(queries)))
print("")
print("==== bCLIP Accuracy ====")
print(accuracy_score(labels, bCLIP.batch_predict(queries)))
print("")
print("==== teCLIP Accuracy ====")
print(accuracy_score(labels, teCLIP.batch_predict(queries)))
