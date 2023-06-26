# OSBC

This is the official implementation repository of the paper "Does Text Matter? Extending OCR with TrOCR and NLP for Image Classification and Retrieval".

![](/OSBC.png)

OSBC (OCR Sentence BERT CLIP) is a novel architecture which extends [CLIP](https://github.com/openai/CLIP) with a text extraction pipeline composed of an OCR model ([TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr) or [PyTesseract](https://github.com/madmaze/pytesseract)) and [SBERT](https://www.sbert.net/). OSBC focuses on leveraging inner text as an additional feature for image classification and retrieval.

## Setup

### Environment
For starters, we need to recreate the environment. We used a Linux machine for development, so other OS users might need to make some workarounds.

To install all the dependencies:

```
conda env create -f environment.yml
```

and don't forget to add the spacy english dictionary:

```
python -m spacy download en_core_web_sm
```

Depending on your OS, PyTesseract will have to be installed in different ways. Refer to [their repo](https://github.com/madmaze/pytesseract) for more information.

### Data

We use the following datasets: [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k), [Standard OCR](https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset), [MNIST](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html), [CIFAR-10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10), and a custom Dilbert dataset.

The MNIST and CIFAR-10 datasets will be automatically downloaded the first time you run the pipeline, as we used the online torchvision datasets.

For all the other datasets, you have to manually download them in the data directory and refactor them. The data directory should look something like this:

```
.
├── characters
│   ├── train
│   └── validation
├── dilbert
│   ├── captions.csv
│   └── Images
└── flickr8k
    ├── captions.csv
    └── Images
```

The Characters dataset is a subset of the Standard OCR dataset, which only includes the english letters folders.
The Dilbert dataset is avaliable upon request.

## Evaluate

To evaluate one of the pipelines we use in the paper, run the following command:

```
python main.py {task name} {dataset name} {model flag} {optional model versions}
```

For example, to evaluate a custom OSBC model on characters:

```
python main.py classification characters --eval_osbc=True --clip_model="openai/clip-vit-base-patch32" --ocr_model="microsoft/trocr-base-printed"
```

The list of all possible parameters:

```
task: classification, retrieval 
dataset: characters, mnist, cifar, flickr8k, dilbert 
--eval_osbc: Bool (default: False)
--eval_clip: Bool (default: False)
--eval_os: Bool (default: False)
--clip_model: openai/clip-vit-base-patch16, openai/clip-vit-base-patch32, openai/clip-vit-large-patch14 
--ocr_model: microsoft/trocr-base-printed, microsoft/trocr-base-handwritten
```

You must use the right task and dataset combination, and choose at least one model flag.

## Finetuning CLIP

Information on finetuning coming soon.

## Contributors

This repo was created by [Jordan Sassoon](https://github.com/jordisassoon)
For any questions, feel free to reach out.