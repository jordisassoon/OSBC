from models.nCLIPRetriever import nCLIP
from models.bCLIPRetriever import bCLIP
from sklearn.metrics import accuracy_score
from PIL import Image

embeddings = []
bCLIP = bCLIP()
nCLIP = nCLIP()

text = "a comic of Dilbert"
image = Image.open("Demo.jpg")

embeddings.append(bCLIP.forward(image=image, description=text))

text = "a comic of Dilbert and his friend"
image = Image.open("Demo2.jpg")

embeddings.append(bCLIP.forward(image=image, description=text))

text = "a diagram"
image = Image.open("CLIP.png")

embeddings.append(bCLIP.forward(image=image, description=text))

queries = ["A comic of Dilbert talking about passwords", "A comic of Dilbert talking about blockchain", "A diagram"]

labels = [0, 1, 2]
print("")
print("==== CLIP Accuracy ====")
print(accuracy_score(labels, nCLIP.batch_predict(queries, embeddings)))
print("")
print("==== bCLIP Accuracy ====")
print(accuracy_score(labels, bCLIP.batch_predict(queries, embeddings)))
