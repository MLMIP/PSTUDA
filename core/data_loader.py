from munch import Munch
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from core.kidney_dataset import *
from torchvision.transforms import functional as F
import torchvision.transforms as T
import random


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


class RandomResize:
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img):
        size = random.randint(self.min_size, self.max_size)
        img = F.resize(img, [size])
        return img


def get_loader_kidney(root, data_txt_file, img_size=256, batch_size=8, prob=0.5, num_workers=4):

    if img_size == 256:
        min_size = int(0.5 * 512)
        max_size = int(0.6 * 512)
    elif img_size == 512:
        min_size = int(1.0 * 512)
        max_size = int(1.1 * 512)
    else:
        raise NotImplementedError

    transform = T.Compose([T.ToPILImage(),
                           RandomResize(min_size, max_size),
                           T.RandomHorizontalFlip(prob),
                           T.RandomVerticalFlip(prob),
                           T.RandomCrop(img_size)])

    dataset = KidneyDataset(data_root=root, data_txt_file=data_txt_file, transforms=transform)

    sampler = _make_balanced_sampler(dataset.targets)

    return DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)


class InputFetcher:
    def __init__(self, loader):
        self.loader = loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def __next__(self):
        x, y = self._fetch_inputs()
        inputs = Munch(x_src=x, y_src=y)

        return Munch({k: v.to(self.device) for k, v in inputs.items()})

