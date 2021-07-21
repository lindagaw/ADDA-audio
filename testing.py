import os
import numpy as np
import torch
import tensorflow as tf
from tensorflow import keras

from load_1st_half_chopped_source_model import load_chopped_source_model
from load_2nd_half_chopped_source_model import load_second_half_chopped_source_model

from models import Discriminator, GalateaEncoder, GalateaClassifier

from models import AurielEncoder, BeatriceEncoder, CielEncoder, DioneEncoder
from models import AurielClassifier, BeatriceClassifier, CielClassifier, DioneClassifier

conv1 = 'snapshots_conv_1'
conv2 = 'snapshots_conv_2'
conv3 = 'snapshots_conv_3'
conv4 = 'snapshots_conv_4'

tgt_classifier = 'ADDA-target-classifier-final.pt'
src_classifier = 'ADDA-source-classifier-final.pt'
tgt_encoder = 'ADDA-target-encoder-final.pt'
src_encoder = 'ADDA-source-encoder-final.pt'
critic = 'ADDA-critic-final.pt'

convs = [conv1, conv2, conv3, conv4]
tgt_classifiers = []
tgt_encoders = []
critics = []

up_until_conv1 = load_chopped_source_model(1)
up_until_conv2 = load_chopped_source_model(2)
up_until_conv3 = load_chopped_source_model(3)
up_until_conv4 = load_chopped_source_model(4)
after_conv4 = load_second_half_chopped_source_model(4)

model = tf.keras.models.load_model('..//model.hdf5')
model.summary()

for conv in convs:
    tgt_classifiers.append(torch.load(os.path.join(conv, tgt_classifier)))
    tgt_encoders.append(torch.load(os.path.join( conv, tgt_encoder)))
    critics.append(torch.load(os.path.join(conv, critic)))

testing_xs = np.load(os.path.join( '..', '..', 'Datasets', 'CONFLICT', 'conflict_testing_xs.npy'))
testing_ys = np.load(os.path.join('..', '..', 'Datasets', 'CONFLICT', 'conflict_testing_ys.npy'))

def get_models(conv):
    if conv == 0:
        src_encoder_net = GalateaEncoder()
        src_classifier_net = GalateaClassifier()
        tgt_classifier_net = GalateaClassifier()
        tgt_encoder_net = GalateaEncoder()

    elif conv == 1:
        src_encoder_net = AurielEncoder()
        src_classifier_net = AurielClassifier()
        tgt_classifier_net = AurielClassifier()
        tgt_encoder_net = AurielEncoder()
    elif conv == 2:
        src_encoder_net = BeatriceEncoder()
        src_classifier_net = BeatriceClassifier()
        tgt_classifier_net = BeatriceClassifier()
        tgt_encoder_net = BeatriceEncoder()
    elif conv == 3:
        src_encoder_net = CielEncoder()
        src_classifier_net = CielClassifier()
        tgt_classifier_net = CielClassifier()
        tgt_encoder_net = CielEncoder()
    else:
        src_encoder_net = DioneEncoder()
        src_classifier_net = DioneClassifier()
        tgt_classifier_net = DioneClassifier()
        tgt_encoder_net = DioneEncoder()

