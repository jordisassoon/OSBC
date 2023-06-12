import torch
import numpy as np
from tqdm import tqdm
from models.components.ocr import OCRComponent
from models.components.sbert import SBERTComponent
from models.components.clip_model import CLIPModelComponent
from models.components.clip_vision import CLIPVisionComponent
from models.components.clip_text import CLIPTextComponent

class OSBC:
    def __init__(self, ocr_model_name, sbert_model_name, clip_model_name) -> None:
        self.ocr = OCRComponent(ocr_model_name)
        self.sbert = SBERTComponent(sbert_model_name)
        self.clip = CLIPModelComponent(clip_model_name)
        self.clip_vision = CLIPVisionComponent(clip_model_name)
        self.clip_text = CLIPTextComponent(clip_model_name)

    def forward_classification(self, dataloader, raw_labels, clip_labels, threshold):

        processed_labels = [self.sbert.process_text(text, stopwords=True) for text in raw_labels]
        sbert_labels = self.sbert.encode_text(processed_labels)

        predictions = np.array([])

        for batch in tqdm(dataloader):
            images, _ = batch

            extracted_texts = self.ocr.forward(images=images)

            processed_texts = [self.sbert.process_text(text, stopwords=True) for text in extracted_texts]
            encoded_texts = self.sbert.encode_text(processed_texts)
            sbert_output = self.sbert.similarity_score(encoded_texts, sbert_labels)

            clip_output = self.clip.forward(images=images, texts=clip_labels).cpu() / 100

            m = torch.nn.Threshold(threshold, 0)
            sbert_preds_clipped = m(sbert_output)
            clip_preds_clipped = m(clip_output)

            predictions = np.append(predictions, (sbert_preds_clipped + clip_preds_clipped).argmax(dim=1).numpy())
        
        return predictions
    
    def forward_retrieval(self, dataloader, threshold):

        sbert_embeddings = None
        image_embeddings = torch.tensor([]).to(self.clip.device)
        predictions = np.array([])
        
        for batch in tqdm(dataloader):
            images, _ = batch
            
            extracted_texts = self.ocr.forward(images=images)
            processed_texts = [self.sbert.process_text(text) for text in extracted_texts]
            text_embeddings = self.sbert.encode_text(processed_texts)
            
            if sbert_embeddings is None:
                sbert_embeddings = text_embeddings
            else:
                sbert_embeddings = np.append(sbert_embeddings, text_embeddings, axis=0)

            embeddings = self.clip_vision.forward(images=images)
            image_embeddings = torch.cat((image_embeddings, embeddings), 0)

        image_embeddings = image_embeddings.T

        for batch in tqdm(dataloader):
            _, captions = batch

            for caption_list in captions:
                processed_texts = [self.sbert.process_text(text) for text in caption_list]
                os_text_embeddings = self.sbert.encode_text(processed_texts)
                os_similarity_scores = self.sbert.similarity_score(os_text_embeddings, sbert_embeddings)

                clip_text_embeddings = self.clip_text.forward(text=caption_list)
                clip_similarity_scores = torch.matmul(clip_text_embeddings, image_embeddings).cpu().numpy() / 100

                os_similarity_scores[processed_texts == ""][os_similarity_scores[processed_texts == ""]!=0] = 0
                os_similarity_scores[os_similarity_scores < threshold] = 0

                predictions = np.append(predictions, np.argmax((os_similarity_scores + clip_similarity_scores), axis=1))

        return predictions
