import pickle
import torch
import numpy as np
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader
import sound_params as params

import os
import gzip
from torchvision import datasets, transforms

class CONFLICT(data.Dataset):

    def __init__(self, root, train=True, transform=None, download=False):
        """Init USPS dataset."""
        # init params
        self.root = 'D://Datasets//CONFLICT//'
        self.filename = "conflict.pkl"
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None

        # download dataset.
        if download:

            pre_process = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(
                                                  mean=params.dataset_mean,
                                                  std=params.dataset_std)])

            pre_process =  transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

            conflict_npy_path = 'D://Datasets//CONFLICT//'

            print('Started loading training data...')
            xs_train = np.load(conflict_npy_path + 'conflict_training_xs.npy')
            ys_train = np.load(conflict_npy_path + 'conflict_training_ys.npy')
            xs_test = np.load(conflict_npy_path + 'conflict_testing_xs.npy')
            ys_test = np.load(conflict_npy_path + 'conflict_testing_ys.npy')

            x = torch.Tensor(np.vstack((xs_train, xs_test)))
            y = torch.Tensor(np.vstack((ys_train, ys_test)))

            conflict_dataset = TensorDataset(x, y) # create your datset)
            torch.save(conflict_dataset, conflict_npy_path + 'conflict.pkl')

            data_set = torch.load(conflict_npy_path + 'conflict.pkl')

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        self.train_data, self.train_labels = self.load_samples()

        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], ::]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]
        #self.train_data *= 255.0
        #self.train_data = self.train_data.transpose(
        #    (0, 2, 3, 1))

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.train_data[index, ::], self.train_labels[index]
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([np.int64(label).item()])
        # label = torch.FloatTensor([label.item()])
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(self.root + self.filename)


    def load_samples(self):
        """Load sample images from dataset."""
        filename = self.root + self.filename

        f = filename
        data_set = torch.load(f)

        for item in enumerate(data_set):
            print(len(item))
        '''
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        '''
        return images, labels

def get_conflict(train):

    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)])
    pre_process =  transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    conflict_dataset = CONFLICT(root=params.data_root,
                        train=train,
                        transform=pre_process,
                        download=True)

    conflict_data_loader = torch.utils.data.DataLoader(
        dataset=conflict_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return conflict_data_loader
