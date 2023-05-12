import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class ImageLoader:
    def __init__(self, images_dir):
        self.dataset = datasets.ImageFolder(root=images_dir)

    def get_loader(self, batch_size=64, num_workers=0):
        return DataLoader(dataset=self.dataset, batch_size=batch_size, num_workers=num_workers)
