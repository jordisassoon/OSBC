import numpy as np
from tqdm import tqdm
from models.components.clip import CLIPComponent

class CLIP:
    def __init__(self, clip_model_name) -> None:
        self.clip = CLIPComponent(clip_model_name)

    def forward_classification(self, dataloader, clip_labels):

        predictions = np.array([])

        for batch in tqdm(dataloader):
            images, _ = batch

            clip_output = self.clip.forward(images=images, texts=clip_labels).cpu()

            predictions = np.append(predictions, clip_output.argmax(dim=1).numpy())
        
        return predictions
    
    def forward_retrieval():
        return None
