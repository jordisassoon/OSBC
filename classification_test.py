path = './data/'
images_dir=path+'characters/validation'

print("loading images...")

from dataloaders.image_loader import ImageLoader

image_loader = ImageLoader(images_dir=images_dir, image_size=(32, 32))
dataloader = image_loader.get_loader(batch_size=8)

print("images loaded, preparing data...")

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
            'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
            'W', 'X', 'Y', 'Z']
template = "an image of the letter {}"

formatted_labels = []
truth = []

for label in labels:
    formatted_labels.append(template.format(label))

for datapoint in image_loader.dataset:
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