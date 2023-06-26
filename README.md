# OSBC

This is the official implementation repository of the paper "Does Text Matter? Extending OCR with TrOCR and NLP for Image Classification and Retrieval".

![](/OSBC.png)

OSBC (OCR Sentence BERT CLIP) is a novel architecture which extends [CLIP](https://github.com/openai/CLIP) with a text extraction pipeline composed of an OCR model ([TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr) or [PyTesseract](https://github.com/madmaze/pytesseract)) and [SBERT](https://www.sbert.net/). OSBC focuses on leveraging inner text as an additional feature for image classification and retrieval.

## Setup

## Evaluate