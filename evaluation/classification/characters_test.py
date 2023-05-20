from evaluation.classification.classification_test import ClassificationTest
from models.classifiers.combined_classifier import Comparator
from dataloaders.image_loader import ImageLoader

image_loader = ImageLoader(images_dir='../../data/characters/training_data')

images = []
truth = []
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
          'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
          'W', 'X', 'Y', 'Z']
template = "an image of the letter: {}"

for i, label in enumerate(labels):
    labels[i] = format(label, template)

for datapoint in image_loader.dataset:
    image, label = datapoint
    images.append(image)
    truth.append(label)

comparator = Comparator(clip_model="RN50")
tester = ClassificationTest(comparator=comparator, queries=images, truth=truth)
tester.embed(labels=labels, template=template)
tester.test()
