from ..tester import Test
from tqdm import tqdm


class ClassificationTest(Test):
    def __init__(self, comparator, queries, truth):
        super().__init__(comparator, queries, truth)
        self.embeddings = []

    def embed(self, labels, template):
        for label in tqdm(labels):
            self.embeddings.append(self.comparator.forward(template, label))
        self.comparator.set_embeddings(self.embeddings)
