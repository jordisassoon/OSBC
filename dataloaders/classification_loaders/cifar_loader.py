from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CIFARLoader:
    def __init__(self, image_size):
        self.dataset = datasets.CIFAR10(
            root='./data', 
            train = False, 
            download=True,
            transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.PILToTensor()
                ])
        )

    def get_loader(self, batch_size=64, num_workers=0):
        return DataLoader(dataset=self.dataset, batch_size=batch_size, num_workers=num_workers)