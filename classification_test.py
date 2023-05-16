from models.classifiers.combined_classifier import Comparator
from dataloaders.image_loader import ImageLoader
from validation.score import Score

image_loader = ImageLoader(images_dir='data/characters/testing_data')

images = []
truth = []
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
          'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
          'W', 'X', 'Y', 'Z']
template = "an image of the letter: "

for datapoint in image_loader.dataset:
    image, label = datapoint
    images.append(image)
    truth.append(label)

comparator = Comparator()
scorer = Score(images, truth)

embeddings = []
for label in labels:
    embeddings.append(comparator.forward(template, label))
comparator.set_embeddings(embeddings)

scores = scorer.score_comparator(comparator)
print("OCR Classifier score: " + str(scores[0]))
print("CLIP Classifier score: " + str(scores[1]))
print("BERT-CLIP Classifier score: " + str(scores[2]))
print("teCLIP Classifier score: " + str(scores[3]))
