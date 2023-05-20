from validation.score import Score


class Test:
    def __init__(self, comparator, queries, truth):
        self.comparator = comparator
        self.queries = queries
        self.truth = truth

    def test(self):
        scorer = Score(self.queries, self.truth)
        scores = scorer.score_comparator(self.comparator)

        print("OCR Classifier score: " + str(scores[0]))
        print("CLIP Classifier score: " + str(scores[1]))
        print("BERT-CLIP Classifier score: " + str(scores[2]))
        print("teCLIP Classifier score: " + str(scores[3]))
