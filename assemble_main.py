from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

import tensorflow as tf
from tensorflow import keras

import numpy as np
import torch

import sound_params as params
from core import eval_src, eval_tgt, train_src, train_tgt, eval_tgt_ood, train_tgt_classifier
from models import Discriminator, GalateaEncoder, GalateaClassifier

from models import AurielEncoder, BeatriceEncoder, CielEncoder, DioneEncoder
from models import AurielClassifier, BeatriceClassifier, CielClassifier, DioneClassifier
from utils import get_data_loader, init_model, init_random_seed

from source_classifier import load_chopped_source_model, load_second_half_chopped_source_model
from sklearn.metrics import f1_score

import sys

def trio(tgt_classifier_net, tgt_encoder_net, src_dataset, tgt_dataset, conv):
    print('loading pretrained trio after conv ' + str(conv) + '...')

    tgt_classifier = init_model(net=tgt_classifier_net,
                                restore= str(conv) + "_snapshots/" + \
                                src_dataset + "-ADDA-target-classifier-final.pt")
    tgt_encoder = init_model(net=tgt_encoder_net,
                             restore= str(conv) + "_snapshots/" + \
                            tgt_dataset + "-ADDA-target-classifier-final.pt")

    critic = init_model(Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims),
                        restore= str(conv) + "_snapshots/" + \
                        tgt_dataset + "-ADDA-target-classifier-final.pt")

    return tgt_classifier, tgt_encoder, critic

if __name__ == '__main__':

    tgt_classifier_1, tgt_encoder_1, critic_1 = \
        trio(AurielClassifier(), AurielEncoder(), 'CONV_1_ACTIVATIONS', 'CONV_1_ACTIVATIONS', 1)
    tgt_classifier_2, tgt_encoder_2, critic_2 = \
        trio(BeatriceClassifier(), BeatriceEncoder(), 'CONV_2_ACTIVATIONS', 'CONV_2_ACTIVATIONS', 2)
    tgt_classifier_3, tgt_encoder_3, critic_3 = \
        trio(CielClassifier(), CielEncoder(), 'CONV_3_ACTIVATIONS', 'CONV_3_ACTIVATIONS', 3)
    tgt_classifier_4, tgt_encoder_4, critic_4 = \
        trio(DioneClassifier(), DioneEncoder(), 'CONV_4_ACTIVATIONS', 'CONV_4_ACTIVATIONS', 4)

    tgt_classifiers = [tgt_classifier_1, tgt_classifier_2, tgt_classifier_3, tgt_classifier_4]
    tgt_encoders = [tgt_encoder_1, tgt_encoder_2, tgt_encoder_3, tgt_encoder_4]
    critics = [critic_1, critic_2, critic_3, critic_4]

    xs_testing = np.load('..//..//Datasets//CONFLICT//conflict_testing_xs.npy')
    ys_testing = np.load('..//..//Datasets//CONFLICT//conflict_testing_ys.npy')

    ys_testing = [np.argmax(val) for val in ys_testing]


    after_conv1 = load_chopped_source_model(conv=1)
    after_conv2 = load_chopped_source_model(conv=2)
    after_conv3 = load_chopped_source_model(conv=3)
    after_conv4 = load_chopped_source_model(conv=4)

    convs = [after_conv1, after_conv2, after_conv3, after_conv4]
    original_convs = tf.keras.models.load_model('..//model.hdf5')

    y_preds = []

    for x in xs_testing:
        x = np.expand_dims(x, axis=0)

        for i in range(0, 4):

            activation = torch.from_numpy(convs[i].predict(x))
            activation = activation.reshape((activation.shape[0], activation.shape[2], activation.shape[1]))
            encoded = tgt_encoders[i](activation.cuda())
            criticized = critics[i](encoded)
            origin = torch.argmax(criticized.squeeze())
            
            if origin == 1:
                y_pred = torch.argmax(tgt_classifiers[i](encoded))
                y_preds.append(int(y_pred))
                break


    f1 = f1_score(ys_testing, y_preds, average='weighted')
    print('f1 score = {}'.format(f1))




