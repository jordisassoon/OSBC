from models.classifiers.nCLIPClassifier import nCLIPClassifier
from models.classifiers.bCLIPClassifier import bCLIPClassifier
from models.classifiers.OCRClassifier import OCRClassifier
from models.classifiers.teCLIPClassifier import teCLIPClassifier
from dataloaders.image_loader import ImageLoader
from validation.score import Score

image_loader = ImageLoader(images_dir='data/characters/training_data')

images = []
labels = []

for datapoint in image_loader.dataset:
    image, label = datapoint
    images.append(image)
    labels.append(label)

bCLIPClassifier = bCLIPClassifier()
nCLIPClassifier = nCLIPClassifier()
ocrClassifier = OCRClassifier()
teCLIPClassifier = teCLIPClassifier()

scorer = Score(images, labels)

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
          'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
          'w', 'x', 'y', 'z']
template = 'a photo of the letter: "{}".'

embeddings_bclip = []
embeddings_nclip = []
embeddings_ocr = []
embeddings_teclip = []

for label in labels:
    embeddings_ocr.append(ocrClassifier.forward(label))
    embeddings_bclip.append(bCLIPClassifier.forward(label))
    embeddings_nclip.append(nCLIPClassifier.forward(label))
    embeddings_teclip.append(teCLIPClassifier.forward(label))

nCLIPClassifier.set_embeddings(embeddings_nclip)
bCLIPClassifier.set_embeddings(embeddings_bclip)
ocrClassifier.set_embeddings(embeddings_ocr)
teCLIPClassifier.set_embeddings(embeddings_teclip)

oscore = scorer.score(ocrClassifier)
nscore = scorer.score(nCLIPClassifier)
bscore = scorer.score(bCLIPClassifier)
tescore = scorer.score(teCLIPClassifier)

print("OCR Classifier score: " + str(oscore))
print("CLIP Classifier score: " + str(nscore))
print("BERT-CLIP Classifier score: " + str(bscore))
print("teCLIP Classifier score: " + str(tescore))
