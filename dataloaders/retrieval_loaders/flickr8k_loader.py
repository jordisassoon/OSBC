import os
import pandas as pd
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset

class Flickr8kDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_captions = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_captions)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_captions.iloc[idx, 0])
        image = read_image(img_path, mode=ImageReadMode.RGB)
        caption = self.img_captions.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, caption