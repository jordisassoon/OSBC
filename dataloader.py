from dataloaders.retrieval_loaders.flickr8k_loader import Flickr8kDataset
from torch.utils.data import DataLoader
from torchvision import transforms

path = './data/flickr8k/'
captions = path + "captions.csv"
images = path + "Images"

import pandas as pd
pd.read_csv(captions)

dataset = Flickr8kDataset(captions, images, 
                             transform=transforms.Compose([
                                 transforms.Resize((256, 256))
                                 ]))

dataloader = DataLoader(dataset=dataset, batch_size=256, num_workers=0)

for batch in dataloader:
    images, labels = batch
    print(labels)

