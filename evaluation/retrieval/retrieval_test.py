from ..tester import Test
from tqdm import tqdm


class RetrievalTest(Test):
    def __init__(self, comparator, queries, truth):
        super().__init__(comparator, queries, truth)
        self.embeddings = []

    def embed(self, images):
        for image in tqdm(images):
            self.embeddings.append(self.comparator.forward(image))
        self.comparator.set_embeddings(self.embeddings)
