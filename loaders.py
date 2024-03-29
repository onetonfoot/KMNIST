import os
import numpy as np
import torch
from torchvision.datasets.utils import makedir_exist_ok, download_url
from torch.utils.data import Dataset
from functools import partial


def identity(x):
    return x


class FontDs(Dataset):

    def __init__(self, generator, len, tfms=identity):
        self.len = len
        self.generator = generator
        self.char2idx = {char: i for i, char in enumerate(generator.chars)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.tfms = tfms

    def __getitem__(self, idx):
        img, char = self.generator.sample(1)
        img, char = img[0], char[0]
        return self.tfms(img), self.char2idx[char]

    def __len__(self):
        return self.len


class KujuMNIST(Dataset):
    base_filename = 'kmnist-{}-{}.npz'
    data_filepart = 'imgs'
    labels_filepart = 'labels'

    def __init__(self, urls, folder, train=True, download=True, tfms=None):
        self.root = os.path.expanduser(folder)
        self.urls = urls
        if download:
            self.download()

        self.train = train
        train_or_test = "train" if train else "test"

        self.data = np.load(os.path.join(
            self.root, self.base_filename.format(train_or_test, self.data_filepart)))
        self.data = torch.from_numpy(self.data['arr_0'])
        self.targets = np.load(os.path.join(
            self.root, self.base_filename.format(train_or_test, self.labels_filepart)))
        self.targets = torch.from_numpy(self.targets['arr_0'])
        self.tfms = tfms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        cur_data = self.data[index]

        if self.tfms:
            cur_data = self.tfms(cur_data)

        target = int(self.targets[index])
        img, target = cur_data, target

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        makedir_exist_ok(self.root)
        for url in self.urls:
            filename = url.rpartition('/')[-1]
            download_url(url, root=self.root, filename=filename, md5=None)


kmnist10_urls = [
    'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz',
    'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz',
    'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz',
    'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz',
]

kmnist49_urls = [
    'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz',
    'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz',
    'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz',
    'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz'
]

# TODO Add Kanji

KMNIST10 = partial(KujuMNIST, kmnist10_urls)
KMNIST49 = partial(KujuMNIST, kmnist49_urls)
