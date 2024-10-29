import os
import numpy as np

from torch.utils.data import Dataset


idx2domain = {'t1c': 0, 't2fs': 1, 't2h': 2, 'dwi': 3, 't1': 4}
threshold_dict = {'t1c': [-0.7, 8.], 't2fs': [-0.5, 9.], 't2h': [-0.8, 5.], 'dwi': [-0.5, 12.], 't1': [-0.8, 8.]}


class KidneyDataset(Dataset):
    def __init__(self, data_root, data_txt_file=['4:4:1/filter_t1c_train.txt', '4:4:1/filter_t2fs_train.txt'], transforms=None):
        super(KidneyDataset, self).__init__()
        self.root = data_root
        self.threshold = {}
        self.transforms = transforms
        self.samples, self.targets = [], []
        for _, u in enumerate(data_txt_file):
            with open(os.path.join(data_root, u), mode='r') as f:
                tmp = f.read().split('\n')[:-1]
                self.samples += tmp
            self.targets += len(tmp) * [int(idx2domain[u.split('/')[1].split('_')[-2]])]
            self.threshold[int(idx2domain[u.split('/')[1].split('_')[-2]])] = threshold_dict[u.split('/')[1].split('_')[-2]]

        self.samples = list(zip(self.samples, self.targets))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        path, target = self.samples[item]
        img = np.load(os.path.join(self.root, path)).astype(np.float32)
        img[img < self.threshold[target][0]] = self.threshold[target][0]
        img[img > self.threshold[target][1]] = self.threshold[target][1]
        img = 2 * (img - self.threshold[target][0]) / (self.threshold[target][1] - self.threshold[target][0]) - 1

        if self.transforms:
            img = self.transforms(img)

        img = np.array(img)
        img = img.reshape([1, *img.shape])

        return img, target