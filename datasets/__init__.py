
class Dataset:

    def __init__(self):
        pass

    def next_batch(self, batch_size: int = 64, num_images: int = 2, num_texts: int = 5):
        pass

    def next_batch_features(self, batch_size: int = 64, num_images: int = 2, num_texts: int = 5):
        pass

    def sequential_evaluation_batches(self, batch_size: int = 64, num_images: int = 2, num_texts: int = 5):
        pass

    def sequential_evaluation_batches_features(self, batch_size: int = 64, num_images: int = 2, num_texts: int = 5):
        pass


