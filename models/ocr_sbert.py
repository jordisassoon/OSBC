import numpy as np
from tqdm import tqdm
from models.components.ocr import OCRComponent
from models.components.sbert import SBERTComponent

class OS:
    def __init__(self, ocr_model_name, sbert_model_name) -> None:
        self.ocr = OCRComponent(ocr_model_name)
        self.sbert = SBERTComponent(sbert_model_name)

    def forward_classification(self, dataloader, raw_labels):

        processed_labels = [self.sbert.process_text(text, numeric=True, stop_words=True) for text in raw_labels]
        sbert_labels = self.sbert.encode_text(processed_labels)
        predictions = np.array([])

        for batch in tqdm(dataloader):
            images, _ = batch

            extracted_texts = self.ocr.forward(images=images)

            processed_texts = [self.sbert.process_text(text, numeric=True, stop_words=True) for text in extracted_texts]
            encoded_texts = self.sbert.encode_text(processed_texts)
            sbert_output = self.sbert.similarity_score(encoded_texts, sbert_labels)

            predictions = np.append(predictions, sbert_output.argmax(dim=1).numpy())
        
        return predictions
    
    def forward_retrieval(self, dataloader):
        
        sbert_embeddings = None
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
        
        for batch in tqdm(dataloader):
            _, captions = batch

            for caption_list in captions:
                processed_texts = [self.sbert.process_text(text) for text in caption_list]
                text_embeddings = self.sbert.encode_text(processed_texts)
                
                predictions = np.append(predictions, np.argmax(self.sbert.similarity_score(text_embeddings, sbert_embeddings), axis=1))

        return predictions
