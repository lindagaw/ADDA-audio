import os
import torch
import tensorflow as tf
from tensorflow import keras

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
critic = []

model = tf.keras.models.load_model('..//model.hdf5')
model.summary()

for conv in convs:
    tgt_classifiers.append(torch.load(os.path.join('..', conv, tgt_classifier)))
    tgt_encoders.append(torch.load(os.path.join('..', conv, tgt_encoder)))
    critics.append(torch.load(os.path.join('..', conv, critic)))
