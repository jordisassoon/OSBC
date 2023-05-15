import pytesseract


class OCR:
    def __init__(self, config):
        self.config = config
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def extract_text(self, image):
        return pytesseract.image_to_string(image, config=self.config)
