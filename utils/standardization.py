import numpy as np



class Standardization():
    def __init__(self, mean, std):
        if mean is not None and std is not None:
            self.mean = np.array(mean, dtype=np.float32)
            self.std = np.array(std, dtype=np.float32)
        elif mean is None and std is None:
            self.mean = mean
            self.std = std
        else:
            print('error')
    def __call__(self, img, bboxes, labels):
        if self.mean is not None:
            img -= self.mean
            img /= self.std
        return img, bboxes, labels