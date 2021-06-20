import torch
import random
import numpy as np
import torchvision
from torchvision import transforms as T
from torchvision.transforms import functional as TF

# Modified From https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
def pad_if_smaller(img, size, fill=0):
    img_size = (img.size[-2],img.size[-1])
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
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=1.0):
        self.colorjitter = T.ColorJitter(brightness, contrast, saturation, hue); self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
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
        target = TF.resize(target, s, interpolation=T.InterpolationMode.NEAREST)
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

def get_image_with_suggestion(image, bi_tgt, blur_threshold, window_number = 10):
    blurred_tgt = bi_tgt.clone()
    w, h = blurred_tgt.size()
    I = w//window_number
    J = h//window_number
    x = np.random.randint(1, blur_threshold)
    y = np.random.randint(1, blur_threshold)
    for i in range(1, window_number-1):
        for j in range(1, window_number-1):
            for m in range(-1, 2):
                for n in range(-1, 2):
                    blurred_tgt[i*I: (i+1)*I-1, j*J: (j+1)*J-1] = torch.logical_or(blurred_tgt[i*I: (i+1)*I-1, j*J: (j+1)*J-1], blurred_tgt[i*I + m*x: (i+1)*I-1 + m*x, j*J + n*y: (j+1)*J-1 + n*y])
    blurred_tgt = blurred_tgt.unsqueeze(0)
    image_with_suggestion = torch.cat([image, blurred_tgt], dim = 0)
    return image_with_suggestion

def get_bi_tgt(new_tgt, label_list, threshold):
    # return the matrix where "true" replaces all the positions of a random object (object i, 0 <= i <= 19) large enough in the matrix new_tgt
    count = [0 for i in range(max(label_list)+1)]
    large_object_labels = []
    for i in label_list:
        count[i] = (new_tgt == i).sum()
        if count[i] > threshold:
            large_object_labels.append(i)
    # n = count.index(max(count))
    n = random.choice(large_object_labels)
    bi_tgt = (new_tgt == n)
    return bi_tgt.long()

class SuggestionTransform(object):
    def __init__(self, label_list = list(range(1, 20)), threshold = 100, blur_threshold = 20, window_number = 10):
        self.label_list = label_list
        self.threshold = threshold
        self.blur_threshold = blur_threshold
        self.window_number = window_number
    def __call__(self, image, target):
        new_tgt = target.clone()
        bi_tgt = get_bi_tgt(new_tgt, [i+1 for i in range(19)], self.threshold)
        image_with_suggestion = get_image_with_suggestion(image, bi_tgt, self.blur_threshold, self.window_number)
        return image_with_suggestion, bi_tgt

CITYSCAPES_IMAGE_SIZE = (1024,2048)
CITYSCAPES_REDUCED_IMAGE_SIZE = (512,1024)
DEFAULT_TRANSFORM = Compose([
    # ResizeImage(CITYSCAPES_REDUCED_IMAGE_SIZE),
    ToTensor(),
    Normalize((.485, .456, .406), (.229, .224, .225)),
    CityscapesTargetTransform(),
])
AUGMENT_TRANSFORM = Compose([
    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    RandomResizeImage(CITYSCAPES_IMAGE_SIZE, min_scale=0.75, max_scale=1.5),
    RandomGaussianBlur((7,7),(0.1,2.0),p=0.5),
    RandomCrop(768),
    RandomHorizontalFlip(0.5),
    ToTensor(),
    Normalize((.485, .456, .406), (.229, .224, .225)),
    CityscapesTargetTransform(),
    # RandomResizeImage(size=CITYSCAPES_IMAGE_SIZE,min_scale=0.75,max_scale=1.25),
    # ResizeImage(CITYSCAPES_REDUCED_IMAGE_SIZE),
])
SUGGEST_TRANSFORM = Compose([
    ToTensor(),
    Normalize((.485, .456, .406), (.229, .224, .225)),
    CityscapesTargetTransform(),
    SuggestionTransform(),
    ResizeImage(CITYSCAPES_REDUCED_IMAGE_SIZE),
])
def CREATE_SUGGEST_TRANSFORM(threshold = 100, blur_threshold = 20, window_number = 10):
    return Compose([
        ToTensor(),
        Normalize((.485, .456, .406), (.229, .224, .225)),
        CityscapesTargetTransform(),
        SuggestionTransform(threshold=threshold,blur_threshold=blur_threshold,window_number=window_number),
        ResizeImage(CITYSCAPES_REDUCED_IMAGE_SIZE),
    ])