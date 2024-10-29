import random

import numpy as np
import torch
from torchvision.transforms import functional as F
import torchvision.transforms as T


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, [0, 0, padw, padh], fill=fill)

    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class ToPILImage:
    def __call__(self, img, target):
        img = F.to_pil_image(img, None)
        target = F.to_pil_image(target, None)

        return img, target


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        self.max_size = min_size
        if self.max_size is not None:
            self.max_size = max_size

    def __call__(self, img, target):
        size = random.randint(self.min_size, self.max_size)
        img = F.resize(img, [size])
        target = F.resize(target, [size], interpolation=T.InterpolationMode.NEAREST)
        return img, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, img, target):
        if random.random() < self.flip_prob:
            img = F.hflip(img)
            target = F.hflip(target)
        return img, target


class RandomVerticalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, img, target):
        if random.random() < self.flip_prob:
            img = F.vflip(img)
            target = F.vflip(target)
        return img, target


class RandomCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, target):
        img = pad_if_smaller(img, self.crop_size)
        target = pad_if_smaller(target, self.crop_size, fill=255)
        crop_params = T.RandomCrop.get_params(img, (self.crop_size, self.crop_size))
        img = F.crop(img, *crop_params)
        target = F.crop(target, *crop_params)

        return img, target


class ToTensor:
    def __call__(self, img, target):
        img = F.to_tensor(img)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)

        return img, target


class Normalize:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        img = F.normalize(img, mean=self.mean, std=self.std)

        return img, target