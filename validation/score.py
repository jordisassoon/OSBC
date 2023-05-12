from sklearn.metrics import accuracy_score
import numpy as np


class Score:
    def __init__(self, images=None, labels=None):
        self.images = images
        self.labels = labels

    def score(self, model):
        predictions = model.batch_predict(self.images)
        return accuracy_score(self.labels, predictions)
