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