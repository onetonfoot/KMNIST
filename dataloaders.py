import os
import numpy as np
from torchvision.datasets.utils import makedir_exist_ok, download_url
from torch.utils.data import Dataset
import requests
from functools import partial

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm doesn't exist, replace it with a function that does nothing
    def tqdm(x, total, unit): return x
    print('**** Could not import tqdm. Please install tqdm for download progressbars! (pip install tqdm) ****')


def download_list(url_list):
    for url in url_list:
        path = url.split('/')[-1]
        r = requests.get(url, stream=True)
        with open(path, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            print('Downloading {} - {:.1f} MB'.format(path, (total_length / 1024000)))

            for chunk in tqdm(r.iter_content(chunk_size=1024), total=int(total_length / 1024) + 1, unit="KB"):
                if chunk:
                    f.write(chunk)
    print('All dataset files downloaded!')


class KujuMNIST_DS(Dataset):
    base_filename = 'kmnist-{}-{}.npz'
    data_filepart = 'imgs'
    labels_filepart = 'labels'

    def __init__(self, urls, folder, train_or_test='train', download=True, num_classes=10, max_items=None, tfms=None):
        self.root = os.path.expanduser(folder)
        if download:
            self.download()

        self.train = (train_or_test == 'train')

        self.data = np.load(os.path.join(
            self.root, self.base_filename.format(train_or_test, self.data_filepart)))
        self.data = self.data['arr_0']
        self.targets = np.load(os.path.join(
            self.root, self.base_filename.format(train_or_test, self.labels_filepart)))
        self.targets = self.targets['arr_0']
        self.c = num_classes
        self.max_items = max_items
        self.tfms = tfms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        cur_data = np.expand_dims(self.data[index], axis=-1)

        if self.tfms:
            cur_data = self.tfms(cur_data)

        target = int(self.targets[index])
        img, target = cur_data, target

        return img, target

    def __len__(self):
        if self.max_items:
            return self.max_items
        return len(self.data)

    def download(self):
        makedir_exist_ok(self.root)
        for url in self.urls:
            filename = url.rpartition('/')[-1]
            file_path = os.path.join(self.root, filename)
            download_url(url, root=self.root, filename=filename, md5=None)


kmist10_urls = [
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

#TODO Add Kanji

KMNIST10 = partial(KujuMNIST, kmist10_urls)
KMNIST49 = partial(KujuMNIST, kmist49_urls)
