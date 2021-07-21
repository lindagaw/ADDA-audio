import os
import numpy as np
import torch
import tensorflow as tf
from tensorflow import keras

from load_1st_half_chopped_source_model import load_chopped_source_model
from load_2nd_half_chopped_source_model import load_second_half_chopped_source_model

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

model = tf.keras.models.load_model('..//..//model.hdf5')
model.summary()

for conv in convs:
    tgt_classifiers.append(torch.load(os.path.join('..', conv, tgt_classifier)))
    tgt_encoders.append(torch.load(os.path.join('..', conv, tgt_encoder)))
    critics.append(torch.load(os.path.join('..', conv, critic)))

testing_xs = np.load(os.path.join('..', '..', '..', 'Datasets', 'CONFLICT', 'conflict_testing_xs.npy'))
testing_ys = np.load(os.path.join('..', '..', '..', 'Datasets', 'CONFLICT', 'conflict_testing_ys.npy'))

for x in testing_xs:

    y_pred_conv1 = up_until_conv1.predict(x)
    encoded_conv1 = tgt_encoders[0](x)
