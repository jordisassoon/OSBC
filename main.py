import torch
import clip
import os
from torchvision.datasets import MNIST, CIFAR10
from tqdm import tqdm
import pytesseract
import numpy as np
from transformers import BertTokenizer

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\jsass\Tesseract-OCR\tesseract.exe'

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('RN50', device)
tz = BertTokenizer.from_pretrained("bert-base-cased")

# from https://github.com/openai/CLIP/blob/main/data/prompts.md
mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ]
mnist_templates = ['a photo of the number: "{}".', ]
class_map = {'MNIST': mnist_classes}
template_map = {'MNIST': mnist_templates}


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


@torch.no_grad()
def extract_text_features(dataset_name):
    # code borrowed from: https://github.com/openai/CLIP/blob/fcab8b6eb92af684e7ff0a904464be7b99b49b88/notebooks/Prompt_Engineering_for_ImageNet.ipynb
    class_names = class_map[dataset_name]
    templates = template_map[dataset_name]
    model.to(device)
    model.eval()

    zeroshot_weights = []
    for classname in class_names:
        texts = [template.format(classname) for template in templates]
        texts = clip.tokenize(texts).to(device)
        class_embeddings = model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


mnist = MNIST(root=os.path.expanduser("~/.cache"), download=True, train=False)
print(type(mnist))

for dataset in [mnist]:
    # extract image feature, code borrowed from: https://github.com/openai/CLIP#zero-shot-prediction
    image_features = []
    image_labels = []
    for image, class_id in tqdm(dataset):

        cv_img = np.array(image)
        options = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(cv_img, config=options)

        print('\n'+text)
        print(tz.convert_tokens_to_ids(tz.tokenize(text)))
        break
    #     image_input = preprocess(image).unsqueeze(0).to(device)
    #     with torch.no_grad():
    #         image_feature = model.encode_image(image_input)
    #     image_feature /= image_feature.norm()
    #     image_features.append(image_feature)
    #     image_labels.append(class_id)
    # image_features = torch.stack(image_features, dim=1).to(device)
    # image_features = image_features.squeeze()
    #
    # # extract text feature
    # dataset_name = 'MNIST' if dataset == mnist else 'CIFAR10'
    # text_features = extract_text_features(dataset_name)
    #
    # # compute top-1 accuracy
    # logits = (100. * image_features @ text_features).softmax(dim=-1)
    # image_labels = torch.tensor(image_labels).unsqueeze(dim=1).to(device)
    # top1_acc = accuracy(logits, image_labels, (1,))
    # print(f'top-1 accuracy for {dataset_name} dataset: {top1_acc[0]:.3f}')
