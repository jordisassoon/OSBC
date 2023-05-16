class Embedding:
    def __init__(self, image_embedding=None, description_embedding=None, inner_text=None):
        if image_embedding is not None:
            self.image_embedding = image_embedding
        if description_embedding is not None:
            self.description_embedding = description_embedding
        if inner_text is not None:
            self.inner_text = inner_text

    def get_image(self):
        return self.image_embedding

    def get_description(self):
        return self.description_embedding

    def get_inner_text(self):
        return self.inner_text
