import torch
import numpy as np
from tqdm import tqdm
from models.components.clip_model import CLIPModelComponent
from models.components.clip_vision import CLIPVisionComponent
from models.components.clip_text import CLIPTextComponent

class CLIP:
    def __init__(self, clip_model_name) -> None:
        self.clip = CLIPModelComponent(clip_model_name)
        self.clip_vision = CLIPVisionComponent(clip_model_name)
        self.clip_text = CLIPTextComponent(clip_model_name)

    @torch.no_grad()
    def forward_classification(self, dataloader, clip_labels):

        predictions = np.array([])

        for batch in tqdm(dataloader):
            images, _ = batch

            clip_output = self.clip.forward(images=images, texts=clip_labels).cpu()

            predictions = np.append(predictions, clip_output.argmax(dim=1).numpy())
        
        return predictions
    
    def forward_retrieval(self, dataloader):

        image_embeddings = torch.tensor([]).to(self.clip.device)
        predictions = np.array([])
        
        for batch in tqdm(dataloader):
            images, _ = batch

            embeddings = self.clip_vision.forward(images=images)

            image_embeddings = torch.cat((image_embeddings, embeddings), 0)

        image_embeddings = image_embeddings.T

        for batch in tqdm(dataloader):
            _, captions = batch

            for caption_list in captions:
                text_embeddings = self.clip_text.forward(text=caption_list)

                similarity_scores = torch.matmul(text_embeddings, image_embeddings)
                
                predictions = np.append(predictions, similarity_scores.argmax(dim=1).cpu().numpy())

        return predictions
