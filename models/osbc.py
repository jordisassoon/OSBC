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

    @torch.no_grad()
    def forward_classification(self, dataloader, raw_labels, clip_labels):

        processed_labels = [self.sbert.process_text(text, numeric=True, stop_words=True) for text in raw_labels]
        sbert_labels = self.sbert.encode_text(processed_labels)

        predictions = np.array([])

        for batch in tqdm(dataloader):
            images, _ = batch

            extracted_texts = self.ocr.forward(images=images)

            processed_texts = [self.sbert.process_text(text, numeric=True, stop_words=True) for text in extracted_texts]
            encoded_texts = self.sbert.encode_text(processed_texts)
            sbert_output = self.sbert.similarity_score(encoded_texts, sbert_labels)

            sbert_output[sbert_output < 0] = 0

            for i, row in enumerate(sbert_output):
                if extracted_texts[i] == "":
                    row[row != 0] = 0

            clip_output = self.clip.forward(images=images, texts=clip_labels).cpu()

            sbert_preds_clipped = torch.nn.functional.normalize(sbert_output, p=1.0, dim=1)
            clip_preds_clipped = torch.nn.functional.normalize(clip_output, p=1.0, dim=1)

            predictions = np.append(predictions, (sbert_preds_clipped + clip_preds_clipped).argmax(dim=1).numpy())

        return predictions

    @torch.no_grad()
    def forward_retrieval(self, dataloader):

        sbert_embeddings = None
        raw_texts = None
        image_embeddings = torch.tensor([]).to(self.clip.device)
        predictions = np.array([])

        for batch in tqdm(dataloader):
            images, _ = batch
            
            extracted_texts = self.ocr.forward(images=images)
            processed_texts = [self.sbert.process_text(text, numeric=False, stop_words=False) for text in extracted_texts]
            text_embeddings = self.sbert.encode_text(processed_texts)

            if sbert_embeddings is None:
                sbert_embeddings = text_embeddings
                raw_texts = processed_texts
            else:
                sbert_embeddings = np.append(sbert_embeddings, text_embeddings, axis=0)
                raw_texts = np.append(raw_texts, processed_texts, axis=0)

            embeddings = self.clip_vision.forward(images=images)
            image_embeddings = torch.cat((image_embeddings, embeddings), 0)

        image_embeddings = image_embeddings.T

        for batch in tqdm(dataloader):
            _, captions = batch

            for caption_list in captions:
                processed_texts = [self.sbert.process_text(text, numeric=False, stop_words=False) for text in caption_list]
                os_text_embeddings = self.sbert.encode_text(processed_texts)
                os_similarity_scores = self.sbert.similarity_score(os_text_embeddings, sbert_embeddings)

                clip_text_embeddings = self.clip_text.forward(text=caption_list)
                clip_similarity_scores = torch.matmul(clip_text_embeddings, image_embeddings).cpu()

                for i, row in enumerate(os_similarity_scores):

                    if processed_texts[i] == "":
                        row[row > 0] = 0

                    _sum = row.sum()
                    row[row < 0.7] = 0
                    row /= _sum
                    row[raw_texts == ""] = 0

                clip_preds_clipped = torch.nn.functional.normalize(clip_similarity_scores, p=1.0, dim=1).numpy()

                predictions = np.append(predictions, np.argmax((os_similarity_scores * 0.5 + clip_preds_clipped), axis=1))

        return predictions
