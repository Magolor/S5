import torch
import random
import numpy as np
import torchvision
from torchvision import transforms as T
from torchvision.transforms import functional as TF

# Modified From https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
def pad_if_smaller(img, size, fill=0):
    img_size = (img.shape[-2],img.shape[-1])
    min_size = min(img_size)
    if min_size < size:
        ow, oh = img_size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = TF.pad(img, (0, 0, padw, padh), fill=fill)
    return img

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomGaussianBlur(object):
    def __init__(self, kernel_size=(5,5), sigma=(0.1,2.0), p = 0.5):
        self.kernel_size = kernel_size; self.sigma=sigma; self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = TF.gaussian_blur(image, self.kernel_size, self.sigma)
        return image, target

class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.colorjitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, target):
        image = self.colorjitter(image)
        return image, target

class ResizeImage(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = TF.resize(image, self.size, interpolation=T.InterpolationMode.BILINEAR)
        return image, target

class RandomResizeImage(object):
    def __init__(self, size, min_scale, max_scale=1.0):
        self.size = size; self.min_scale = min_scale; self.max_scale = max_scale; self.interval = max_scale-min_scale

    def __call__(self, image, target):
        p = random.random()*self.interval+self.min_scale; s = (int(self.size[0]*p),int(self.size[1]*p))
        image = TF.resize(image, s, interpolation=T.InterpolationMode.NEAREST)
        target = TF.resize(target.unsqueeze(0), s, interpolation=T.InterpolationMode.NEAREST)[0]
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            target = TF.hflip(target)
        return image, target

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = TF.crop(image, *crop_params)
        target = TF.crop(target, *crop_params)
        return image, target

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = TF.center_crop(image, self.size)
        target = TF.center_crop(target, self.size)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = TF.normalize(image, mean=self.mean, std=self.std)
        return image, target
# End From

CITYSCAPES_35_TO_19 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 2, 9: 0, 10: 0, 11: 3, 12: 4, 13: 5, 14: 0, 15: 0, 16: 0, 17: 6, 18: 0, 19: 7, 20: 8, 21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16, 29: 0, 30: 0, 31: 17, 32: 18, 33: 19, -1: 0}
class CityscapesTargetTransform(object):
    def __init__(self):
        pass
    def __call__(self, image, target):
        new_tgt = torch.zeros_like(target)
        for k,v in CITYSCAPES_35_TO_19.items():
            new_tgt[target==k] = v
        return image, new_tgt

CITYSCAPES_IMAGE_SIZE = (1024,2048)
CITYSCAPES_REDUCED_IMAGE_SIZE = (512,1024)
DEFAULT_TRANSFORM = Compose([
    ToTensor(),
    CityscapesTargetTransform(),
    Normalize((.485, .456, .406), (.229, .224, .225)),
    ResizeImage(CITYSCAPES_REDUCED_IMAGE_SIZE)
])
AUGMENT_TRANSFORM = Compose([
    ToTensor(),
    CityscapesTargetTransform(),
    Normalize((.485, .456, .406), (.229, .224, .225)),
    # RandomResizeImage(size=CITYSCAPES_IMAGE_SIZE,min_scale=0.75,max_scale=1.25),
    RandomCrop(768),
    RandomGaussianBlur((5,5),(0.1,2.0),p=0.5),
    ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    RandomHorizontalFlip(0.5),
])