import os
from PIL import Image
from models.nCLIPClassifier import nCLIPClassifier
from models.bCLIPClassifier import bCLIPClassifier
from dataloaders.image_loader import ImageLoader
from validation.score import Score
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

image_loader = ImageLoader(images_dir='archive/data/testing_data')

bCLIPClassifier = bCLIPClassifier()
nCLIPClassifier = nCLIPClassifier()

scorer = Score(image_array, truth)

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
          'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
          'w', 'x', 'y', 'z']
template = 'a photo of the letter: "{}".'

embeddings_bclip = []
embeddings_nclip = []

for label in labels:
    embeddings_bclip.append(bCLIPClassifier.forward(label))
    embeddings_nclip.append(nCLIPClassifier.forward(label))

print(scorer.batch_score(nCLIPClassifier, embeddings_nclip, image_loader.get_loader()))
