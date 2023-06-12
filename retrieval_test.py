from dataloaders.retrieval_loaders.flickr8k_loader import Flickr8kDataset
from torch.utils.data import DataLoader
from torchvision import transforms

print("loading images...")

path = './data/flickr8k/'
captions = path + "captions.csv"
images = path + "Images"

dataset = Flickr8kDataset(captions, images, 
                             transform=transforms.Compose([
                                 transforms.Resize((256, 256))
                                 ]))

print("images loaded, preparing data...")

def collate_fn(list_items):
     x = []
     y = []
     for x_, y_ in list_items:
         x.append(x_)
         y.append(y_)
     return x, y

dataloader = DataLoader(dataset=dataset, batch_size=16, num_workers=0, collate_fn=collate_fn)

print("data prepared, loading models...")

from models.osbc import OSBC

osbc = OSBC(ocr_model_name="microsoft/trocr-base-printed",
             sbert_model_name="all-mpnet-base-v2", 
             clip_model_name="openai/clip-vit-base-patch32")

print("models loaded, running inference...")

predictions = osbc.forward_retrieval(dataloader)

print("scoring...")

labels = dataset.__getlabels__()

from sklearn.metrics import accuracy_score
print("CLIP: " + str(accuracy_score(labels, predictions)))