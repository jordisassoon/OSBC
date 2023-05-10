import os
from PIL import Image
from models.nCLIPClassifier import nCLIPClassifier
from models.bCLIPClassifier import bCLIPClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

image_array = []
truth = []

for i in tqdm(range(26)):
    # get the path/directory
    folder_dir = "C:/Users/jsass/teCLIP/archive/data/testing_data/" + chr(65 + i)
    for image in os.listdir(folder_dir):

        # check if the image ends with png
        if image.endswith(".png"):
            image_array.append(Image.open(folder_dir + "/" + image))
            truth.append(i)

print(len(truth))

bCLIPClassifier = bCLIPClassifier()
nCLIPClassifier = nCLIPClassifier()

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
          'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
          'w', 'x', 'y', 'z']
template = 'a photo of the letter: "{}".'

embeddings_bclip = []
embeddings_nclip = []

for label in labels:
    embeddings_bclip.append(bCLIPClassifier.forward(label))
    embeddings_nclip.append(nCLIPClassifier.forward(label))

print(accuracy_score(bCLIPClassifier.batch_predict(image_array, embeddings_bclip), truth))
print(accuracy_score(nCLIPClassifier.batch_predict(image_array, embeddings_nclip), truth))
