from tensorflow.keras.datasets import mnist
from models.nCLIPClassifier import nCLIP
from models.bCLIPClassifier import bCLIP
from sklearn.metrics import accuracy_score
from PIL import Image

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

bCLIP = bCLIP()
nCLIP = nCLIP()

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
template = 'a photo of the number: "{}".'

embeddings = []
for label in labels:
    embeddings.append(bCLIP.forward(label))

embeddings2 = []
for label in labels:
    embeddings2.append(nCLIP.forward(label))

y_pred = []
y_pred2 = []
val = 100
for i in range(val):
    image = Image.fromarray(X_train[i])
    y_pred.append(bCLIP.predict(image, embeddings))
    y_pred2.append(nCLIP.predict(image, embeddings2))

print(accuracy_score(y_pred, Y_train[:val]))
print(accuracy_score(y_pred2, Y_train[:val]))
