import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os

def get_emotion():

    emotion_npy_path = 'D://Datasets//TRAINING (sound)//'
    xs_train = np.load(emotion_npy_path + 'emotion_training_xs.npy')
    ys_train = np.load(emotion_npy_path + 'emotion_training_ys.npy')
    xs_test = np.load(emotion_npy_path + 'emotion_testing_xs.npy')
    ys_test = np.load(emotion_npy_path + 'emotion_testing_ys.npy')

    xs = np.vstack((xs_train, xs_test))
    ys = np.vstack((ys_train, ys_test))

    emotion_dataset = TensorDataset(torch.Tensor(xs), torch.Tensor(ys)) # create your datset
    emotion_dataloader = DataLoader(emotion_dataset) # create your dataloader

    return emotion_dataloader
