import torch
import numpy as np
from tqdm import tqdm
from models.components.ocr import OCRComponent
from models.components.sbert import SBERTComponent
from models.components.clip import CLIPComponent

class OSBC:
    def __init__(self, ocr_model_name, sbert_model_name, clip_model_name) -> None:
        self.ocr = OCRComponent(ocr_model_name)
        self.sbert = SBERTComponent(sbert_model_name)
        self.clip = CLIPComponent(clip_model_name)

    def forward_classification(self, dataloader, raw_labels, clip_labels, threshold):

        sbert_labels = self.sbert.encode_text(raw_labels)

        predictions = np.array([])

        for batch in tqdm(dataloader):
            images, _ = batch

            extracted_texts = self.ocr.forward(images=images)

            encoded_texts = self.sbert.encode_text(extracted_texts)
            sbert_output = self.sbert.similarity_score(encoded_texts, sbert_labels)

            clip_output = self.clip.forward(images=images, texts=clip_labels).cpu() / 100

            m = torch.nn.Threshold(threshold, 0)
            sbert_preds_clipped = m(sbert_output)
            clip_preds_clipped = m(clip_output)

            predictions = np.append(predictions, (sbert_preds_clipped + clip_preds_clipped).argmax(dim=1).numpy())
        
        return predictions
    
    def forward_retrieval():
        return None