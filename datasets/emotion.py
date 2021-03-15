import sys
sys.path.append("..")

import os
import pickle
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import sound_params as params
def get_emotion():

    emotion_npy_path = 'D://Datasets//TRAINING (sound)//'

    path_emotion_dataloader = emotion_npy_path + 'emotion_dataloader.pkl'
    path_emotion_dataloader_eval = emotion_npy_path + 'emotion_dataloder_eval.pkl'

    if not os.path.isfile(path_emotion_dataloader):

        print('Started loading training data...')
        xs_train = np.load(emotion_npy_path + 'emotion_training_xs.npy')
        ys_train = np.load(emotion_npy_path + 'emotion_training_ys.npy')
        print('Started constructing training set...')

        emotion_dataset = TensorDataset(torch.Tensor(xs_train), torch.Tensor(ys_train)) # create your datset
        emotion_dataloader = DataLoader(emotion_dataset) # create your dataloader
        torch.save(emotion_dataloader, path_emotion_dataloader)
    else:
        emotion_dataloader = pickle.load(open(path_emotion_dataloader, 'rb'))
        print('Finished loading pre-existing training set.')

    if not os.path.isfile(path_emotion_dataloader_eval):

        print('Started loading testing data...')
        xs_test = np.load(emotion_npy_path + 'emotion_testing_xs.npy')
        ys_test = np.load(emotion_npy_path + 'emotion_testing_ys.npy')


        print('Started constructing testing set...')
        emotion_dataset_eval = TensorDataset(torch.Tensor(xs_test), torch.Tensor(ys_test)) # create your datset
        emotion_dataloader_eval = DataLoader(emotion_dataset_eval) # create your dataloader
        torch.save(emotion_dataloader_eval, path_emotion_dataloader_eval)
    else:
        emotion_dataloader_eval = pickle.load(open(path_emotion_dataloader_eval, 'rb'))
        print('Finished loading pre-existing testing set.')

    print('Finished constructing EMOTIONAL training and testing sets.')

    return emotion_dataloader, emotion_dataloader_eval
