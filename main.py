from dataloaders.image_loader import ImageLoader
from evaluation.classification.classification_test import ClassificationTest
from evaluation.retrieval.retrieval_test import RetrievalTest
from models.retrievers.combined_retriever import Comparator as CombinedRetriever
from models.classifiers.combined_classifier import Comparator as CombinedClassifier
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse


def main(args):

    print(args.task + " task started successfully on " + args.dataset)

    path = './data/'
    if args.task == 'retrieval':

        query_texts = []
        truth = []
        images = []

        if args.dataset == 'flicker8k':
            path += 'flicker8k/'

            df = pd.read_csv(path + "captions.csv")
            df = df.dropna()
            filenames = df["image"]
            query_texts = df["caption"].astype("string")

            last_filename = None
            last_index = -1
            for filename in tqdm(filenames):
                if filename == last_filename:
                    truth.append(last_index)
                else:
                    im = Image.open(path + "Images/" + filename)
                    images.append(im)
                    last_index += 1
                    truth.append(last_index)
                    last_filename = filename

        if args.dataset == 'dilbert':
            path += 'dilbert/'

            df = pd.read_csv(path + "annotations.csv")
            df = df.drop(["col1", "col2", "col3"], axis=1)
            df = df.dropna()
            filenames = df["original_filename"]
            query_texts = df["Comics_text_box"].astype("string")

            for i, filename in tqdm(enumerate(filenames)):
                im = Image.open(path + "panels/" + filename)
                images.append(im)
                truth.append(i)

        comparator = CombinedRetriever(clip_model=args.clip_model, st_name=args.bert_model, config=args.ocr_config)
        tester = RetrievalTest(comparator=comparator, queries=query_texts, truth=truth)
        tester.embed(images=images)
        tester.test()

    if args.task == 'classification':

        images = []
        truth = []
        labels = []
        template = ""

        if args.dataset == 'mnist':
            path += 'mnist/'

            df = pd.read_csv(path + "mnist_test.csv")
            df = df.dropna()

            for i, row in tqdm(df.iterrows()):
                im = Image.fromarray(np.array(row[1:]))
                images.append(im)
                truth.append(row[0])

            labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            template = "an image of the number: "

        elif args.dataset == 'characters':
            image_loader = ImageLoader(images_dir='../../data/characters/training_data')

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

        comparator = CombinedClassifier(clip_model=args.clip_model, st_name=args.bert_model, config=args.ocr_config)
        tester = ClassificationTest(comparator=comparator, queries=images, truth=truth)
        tester.embed(labels=labels, template=template)
        tester.test()


if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(description='Evaluate the models on a specific task')
    parser.add_argument('task', type=str, metavar='STRING',
                        help='type of task: retrieval or classification')
    parser.add_argument('dataset', type=str, metavar='STRING',
                        help='name of the dataset')
    parser.add_argument('--ocr_config', type=str, metavar='STRING',
                        help='config for Tesseract', default=None)
    parser.add_argument('--bert_model', type=str, metavar='STRING',
                        help='version of BERT', default='all-mpnet-base-v2')
    parser.add_argument('--clip_model', type=str, metavar='STRING',
                        help='version of CLIP', default='RN50')
    args = parser.parse_args()
    main(args)
