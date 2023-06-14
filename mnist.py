from torchvision import datasets, transforms
from torch.utils.data import DataLoader

image_size=(28, 28)

test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.PILToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
)

dataloader = DataLoader(dataset=test_data, batch_size=8, num_workers=0)

print("images loaded, preparing data...")

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
template = "an image of the digit: {}"

formatted_labels = []
truth = []

for label in labels:
    formatted_labels.append(template.format(label))

for datapoint in test_data:
    _, label = datapoint
    truth.append(label)

print("data prepared, loading models...")

from models.osbc import OSBC
from models.clip import CLIP
from models.ocr_sbert import OS

ocr_model_name = "microsoft/trocr-base-printed"
sbert_model_name = "all-mpnet-base-v2"
clip_model_name = "openai/clip-vit-large-patch14"

# osbc = OSBC(ocr_model_name=ocr_model_name,
#              sbert_model_name=sbert_model_name, 
#              clip_model_name=clip_model_name)

clip = CLIP(clip_model_name=clip_model_name)

# os = OS(ocr_model_name=ocr_model_name,
#         sbert_model_name=sbert_model_name)

print("models loaded, running inference...")

# osbc_predictions = osbc.forward_classification(dataloader=dataloader, raw_labels=labels, clip_labels=formatted_labels)
clip_predictions = clip.forward_classification(dataloader=dataloader, clip_labels=formatted_labels)
# os_predictions = os.forward_classification(dataloader=dataloader, raw_labels=labels)

from sklearn.metrics import accuracy_score

print("scoring...")

# print("OSBC: " + str(accuracy_score(truth, osbc_predictions)))
# print("OCR-SBERT: " + str(accuracy_score(truth, os_predictions)))
print("CLIP: " + str(accuracy_score(truth, clip_predictions)))