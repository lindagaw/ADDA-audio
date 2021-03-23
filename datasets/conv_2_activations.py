import pickle
import torch
import numpy as np
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader
import sound_params as params

import os
import gzip
from torchvision import datasets, transforms

class CONV_2_ACTIVATIONS(data.Dataset):

    def __init__(self, root, train=True, transform=None, download=False, dataset='undefined'):
        """Init USPS dataset."""

        if not (dataset == 'emotion' or dataset == 'conflict'):
            raise Exception("Parameter dataset's value must be 'emotion' or 'conflict', case sensitive.")

        self.root = 'data//UTAH//conv_2_activations//'
        self.training = dataset + "_conv_2_activations.pkl"
        self.testing = dataset + "_conv_2_activations_eval.pkl"
        self.train = train

        self.transform = transform
        self.dataset_size = None

        print('loading training data from ' + self.training)
        print('loading testing data from ' + self.testing)
        # download dataset.
        if download:

            pre_process = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(
                                                  mean=params.dataset_mean,
                                                  std=params.dataset_std)])

            pre_process =  transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

            xs_train = torch.Tensor(np.load(self.root + '2_conv_activations_' + \
                                    dataset + '_train_x.npy'))
            xs_test = torch.Tensor(np.load(self.root + '2_conv_activations_' + \
                                    dataset + '_test_x.npy'))

            if dataset == 'emotion':
                ys_train = torch.Tensor(np.load('data//UTAH//binary_' + dataset + '_training_ys.npy'))
                ys_test = torch.Tensor(np.load('data//UTAH//binary_' + dataset + '_testing_ys.npy'))
            else:
                ys_train = torch.Tensor(np.load('data//UTAH//' + dataset + '_training_ys.npy'))
                ys_test = torch.Tensor(np.load('data//UTAH//' + dataset + '_testing_ys.npy'))

            torch.save(TensorDataset(xs_train, ys_train), self.root + self.training)
            torch.save(TensorDataset(xs_test, ys_test), self.root + self.testing)

            data_set_train = torch.load(self.root + self.training)
            data_set_test = torch.load(self.root + self.testing)

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
        self.train_data = self.train_data.transpose(2, 1)

        print(self.train_data.shape)

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

        label = label.type(torch.LongTensor)
        # label = torch.FloatTensor([label.item()])
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(self.root + self.training) and os.path.exists(self.root + self.testing)


    def load_samples(self):
        """Load sample images from dataset."""
        if self.train:
            f = self.root + self.training
        else:
            f = self.root + self.testing

        data_set = torch.load(f)

        audios = torch.Tensor([np.asarray(audio) for _, (audio, _) in enumerate(data_set)])
        labels = torch.Tensor([np.argmax(np.asarray(label)) for _, (_, label) in enumerate(data_set)])

        self.dataset_size = labels.shape[0]

        return audios, labels

def get_conv_2_activations(train, dataset):

    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)])
    pre_process =  transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    conv_2_activations_dataset = CONV_2_ACTIVATIONS(root=params.data_root,
                        train=train,
                        #transform=pre_process,
                        download=False,
                        dataset=dataset)

    conv_2_activations_data_loader = torch.utils.data.DataLoader(
        dataset=conv_2_activations_dataset,
        batch_size=params.batch_size*2,
        shuffle=False)

    return conv_2_activations_data_loader
