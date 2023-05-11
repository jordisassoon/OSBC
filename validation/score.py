from sklearn.metrics import accuracy_score
import numpy as np


class Score:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def score(self, model, embeddings, labels):
        predictions = model.batch_predict(self.images, embeddings)
        return accuracy_score(self.labels, predictions)

    @staticmethod
    def batch_score(model, embeddings, loader):
        scores = []
        for data in loader:
            images, labels = data
            predictions = model.batch_predict(images, embeddings)
            scores.append(accuracy_score(labels, predictions))
        return np.mean(scores)
