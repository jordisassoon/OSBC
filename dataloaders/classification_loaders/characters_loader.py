from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import os

class CharactersDataset(Dataset):
    def __init__(self, classes, img_dir, transform=None):
        self.classes = classes
        self.img_dir = img_dir
        self.transform = transform
        self.images = []
        self.labels = []
        for i, _class in enumerate(classes):
            path = img_dir + _class
            for image in os.listdir(path=path):
                # check if the image ends with png
                if (image.endswith(".png")):
                    self.images.append(Image.open(f"{path}/{image}").convert("RGB"))
                    self.labels.append(i)
    
    def __len__(self):
        return len(self.labels)
    
    def __getlabels__(self):
        return self.labels

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image, f"an image of the letter {chr(ord('A') + self.labels[idx])}"

class CharactersLoader:
    def __init__(self, images_dir, image_size):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.float())
            ])
        self.classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.dataset = CharactersDataset(classes=self.classes, img_dir=images_dir, transform=self.transform)

    def get_loader(self, batch_size=64, num_workers=0):
        return DataLoader(dataset=self.dataset, batch_size=batch_size, num_workers=num_workers)