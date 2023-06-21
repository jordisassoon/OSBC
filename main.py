from dataloaders.classification_loaders.characters_loader import CharactersLoader
from dataloaders.retrieval_loaders.flickr8k_loader import Flickr8kDataset
from dataloaders.classification_loaders.mnist_loader import MNISTLoader
from dataloaders.classification_loaders.cifar_loader import CIFARLoader
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from models.osbc import OSBC
from models.clip import CLIP
from models.ocr_sbert import OS
import argparse


def main(args):

    data_path = "./data/"

    if args.task == "classification":
        print("classification task started succesfully")

        raw_labels = []
        formatted_labels = []
        ground_truth = []
        dataloader = None

        if args.dataset == "mnist":
            print("running classification on mnist")
            print("loading images...")

            test_data = MNISTLoader(image_size=(28, 28))
            dataloader = test_data.get_loader(batch_size=8)
            
            print("images loaded, preparing labels...")

            raw_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            template = "an image of the digit: {}"

            for label in raw_labels:
                formatted_labels.append(template.format(label))

            for datapoint in test_data.dataset:
                _, label = datapoint
                ground_truth.append(label)

        if args.dataset == "cifar":
            print("running classification on cifar")
            print("loading images...")

            test_data = CIFARLoader(image_size=(32, 32))
            dataloader = test_data.get_loader(batch_size=8)
            
            print("images loaded, preparing labels...")

            raw_labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            template = "an image of a {}"

            for label in raw_labels:
                formatted_labels.append(template.format(label))

            for datapoint in test_data.dataset:
                _, label = datapoint
                print(label)
                ground_truth.append(label)

        elif args.dataset == "characters":
            print("running classification on characters")
            print("loading images...")

            data_path += "characters/validation/"

            image_loader = CharactersLoader(images_dir=data_path, image_size=(16, 16))
            dataloader = image_loader.get_loader(batch_size=8)

            print("images loaded, preparing labels...")

            raw_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                          'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                          'W', 'X', 'Y', 'Z']
            template = "an image of the letter {}"

            for label in raw_labels:
                formatted_labels.append(template.format(label))

            ground_truth = image_loader.dataset.labels
   
        else:
            print("dataset \"" + args.dataset + "\" not found.")
            return
        
        print("data prepared, loading models...")

        if args.eval_clip:
            clip = CLIP(clip_model_name=args.clip_model)
            clip_predictions = clip.forward_classification(dataloader=dataloader, clip_labels=formatted_labels)
            print(accuracy_score(ground_truth, clip_predictions))
        
        elif args.eval_osbc:
            osbc = OSBC(ocr_model_name=args.ocr_model, sbert_model_name="all-mpnet-base-v2", clip_model_name=args.clip_model)
            osbc_predictions = osbc.forward_classification(dataloader=dataloader, raw_labels=raw_labels, clip_labels=formatted_labels)
            print(accuracy_score(ground_truth, osbc_predictions))

        elif args.eval_os:
            os = OS(ocr_model_name=args.ocr_model, sbert_model_name="all-mpnet-base-v2")
            os_predictions = os.forward_classification(dataloader=dataloader, raw_labels=raw_labels)
            print(accuracy_score(ground_truth, os_predictions))
        
        else:
            print("please select a model to evaluate: clip, osbc, or ocr-sbert")
            return

    elif args.task == "retrieval":
        print("retrieval task started succesfully")

        ground_truth = []
        dataloader = None

        if args.dataset == "flickr8k":
            print("running retrieval on flickr8k")
            print("loading images...")

            data_path += 'flickr8k/'
            captions = data_path + "captions.csv"
            images = data_path + "Images"

            dataset = Flickr8kDataset(captions, images, 
                                        transform=transforms.Compose([
                                            transforms.Resize((384, 384))
                                            ]))

            def collate_fn(list_items):
                x = []
                y = []
                for x_, y_ in list_items:
                    x.append(x_)
                    y.append(y_)
                return x, y
            
            dataloader = DataLoader(dataset=dataset, batch_size=16, num_workers=0, collate_fn=collate_fn)

            print("images loaded, preparing labels...")

            ground_truth = dataset.__getlabels__()

        else:
            print("dataset \"" + args.dataset + "\" not found.")
            return

        print("data prepared, loading models...")

        if args.eval_clip:
            clip = CLIP(clip_model_name=args.clip_model)
            clip_predictions = clip.forward_retrieval(dataloader=dataloader)
            print(accuracy_score(ground_truth, clip_predictions))
        
        if args.eval_osbc:
            osbc = OSBC(ocr_model_name=args.ocr_model, sbert_model_name="all-mpnet-base-v2", clip_model_name=args.clip_model)
            osbc_predictions = osbc.forward_retrieval(dataloader=dataloader)
            print(accuracy_score(ground_truth, osbc_predictions))

        if args.eval_os:
            os = OS(ocr_model_name=args.ocr_model, sbert_model_name="all-mpnet-base-v2")
            os_predictions = os.forward_retrieval(dataloader=dataloader)
            print(accuracy_score(ground_truth, os_predictions))

        else:
            print("please select a model to evaluate: clip, osbc, or ocr-sbert")
            return

    else:
        print("task \"" + args.task + "\" not found.")
        return

if __name__ == '__main__':
    """Entry Point"""
    parser = argparse.ArgumentParser(description='Evaluate the models on a specific task')
    parser.add_argument('task', type=str, metavar='STRING',
                        help='type of task: retrieval or classification')
    parser.add_argument('dataset', type=str, metavar='STRING',
                        help='name of the dataset')
    parser.add_argument('--eval_clip', type=bool, metavar='BOOL',
                        help='evaluate CLIP pipeline', default=False)
    parser.add_argument('--eval_osbc', type=bool, metavar='BOOL',
                        help='evaluate OSBC pipeline', default=False)
    parser.add_argument('--eval_os', type=bool, metavar='BOOL',
                        help='evaluate OCR-SBERT pipeline', default=False)
    parser.add_argument('--ocr_model', type=str, metavar='STRING',
                        help='config for Tesseract', default='microsoft/trocr-base-printed')
    parser.add_argument('--clip_model', type=str, metavar='STRING',
                        help='version of CLIP', default='openai/clip-vit-base-patch32')
    parser.add_argument('--save', type=bool, metavar='BOOL',
                        help='save the model, task, and accuracy', default=False)
    args = parser.parse_args()
    main(args)