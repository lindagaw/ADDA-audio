import sys

import os
import pickle
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

import conflict

def download_conflict():
    '''
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)])
    '''
    pre_process =  transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    conflict_npy_path = 'D://Datasets//CONFLICT//'

    print('Started loading training data...')
    xs_train = np.load(conflict_npy_path + 'conflict_training_xs.npy')
    ys_train = np.load(conflict_npy_path + 'conflict_training_ys.npy')
    xs_test = np.load(conflict_npy_path + 'conflict_testing_xs.npy')
    ys_test = np.load(conflict_npy_path + 'conflict_testing_ys.npy')

    x = torch.Tensor(np.vstack((xs_train, xs_test)))
    y = torch.Tensor(np.vstack((ys_train, ys_test)))

    conflict_dataset = CONFLICT(TensorDataset(x, y)) # create your datset)

    print(type(conflict_dataset))

    torch.save(conflict_dataset, conflict_npy_path + 'conflict.pkl')

    data_set = pickle.load(open(conflict_npy_path + 'conflict.pkl', "rb" ))

    print(type(data_set))

download_conflict()
