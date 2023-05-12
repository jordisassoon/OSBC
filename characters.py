from models.nCLIPClassifier import nCLIPClassifier
from models.bCLIPClassifier import bCLIPClassifier
from models.OCRClassifier import OCRClassifier
from dataloaders.image_loader import ImageLoader
from validation.score import Score

image_loader = ImageLoader(images_dir='archive/data/testing_data')

images = []
labels = []

for datapoint in image_loader.dataset:
    image, label = datapoint
    images.append(image)
    labels.append(label)

bCLIPClassifier = bCLIPClassifier()
nCLIPClassifier = nCLIPClassifier()
ocrClassifier = OCRClassifier()

scorer = Score(images, labels)

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
          'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
          'w', 'x', 'y', 'z']
template = 'a photo of the letter: "{}".'

embeddings_bclip = []
embeddings_nclip = []
embeddings_ocr = []

for label in labels:
    embeddings_ocr.append(ocrClassifier.forward(label))
    embeddings_bclip.append(bCLIPClassifier.forward(label))
    embeddings_nclip.append(nCLIPClassifier.forward(label))

nCLIPClassifier.set_embeddings(embeddings_nclip)
bCLIPClassifier.set_embeddings(embeddings_bclip)
ocrClassifier.set_embeddings(embeddings_ocr)

oscore = scorer.score(ocrClassifier)
bscore = scorer.score(bCLIPClassifier)
nscore = scorer.score(nCLIPClassifier)

print("OCR Classifier score: " + str(oscore))
print("CLIP Classifier score: " + str(nscore))
print("BERT-CLIP Classifier score: " + str(bscore))
