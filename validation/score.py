from sklearn.metrics import accuracy_score
import numpy as np


class Score:
    def __init__(self, images=None, labels=None):
        self.images = images
        self.labels = labels

    def score(self, model):
        predictions = model.batch_predict(self.images)
        return accuracy_score(self.labels, predictions)

    def score_comparator(self, model):
        predictions = np.array(model.batch_predict(self.images))
        bert = accuracy_score(self.labels, predictions[:, 0])
        nCLIP = accuracy_score(self.labels, predictions[:, 1])
        bCLIP = accuracy_score(self.labels, predictions[:, 2])
        teCLIP = accuracy_score(self.labels, predictions[:, 3])
        return [bert, nCLIP, bCLIP, teCLIP]
