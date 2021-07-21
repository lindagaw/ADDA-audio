import os
import numpy as np
import torch
import tensorflow as tf
from tensorflow import keras

from source_classifier import load_chopped_source_model
from source_classifier import load_second_half_chopped_source_model

from models import Discriminator, GalateaEncoder, GalateaClassifier

from models import AurielEncoder, BeatriceEncoder, CielEncoder, DioneEncoder
from models import AurielClassifier, BeatriceClassifier, CielClassifier, DioneClassifier

import sound_params as params

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
    tgt_classifiers.append(os.path.join(conv, tgt_classifier))
    tgt_encoders.append(os.path.join( conv, tgt_encoder))
    critics.append(os.path.join(conv, critic))

testing_xs = np.load(os.path.join( '..', '..', 'Datasets', 'CONFLICT', 'conflict_testing_xs.npy'))
testing_ys = np.load(os.path.join('..', '..', 'Datasets', 'CONFLICT', 'conflict_testing_ys.npy'))


tgt_classifier_conv1 = AurielClassifier()
tgt_encoder_conv1 = AurielEncoder()
critic_conv1 = Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims)
tgt_classifier_conv1.load_state_dict(torch.load(tgt_classifiers[1]), strict=False)
critic_conv1.load_state_dict(torch.load(critics[1]),strict=False)
tgt_encoder_conv1.load_state_dict(torch.load(tgt_encoders[1]), strict=False)


tgt_classifier_conv2 = BeatriceClassifier()
tgt_encoder_conv2 = BeatriceEncoder()
critic_conv2 = Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims)
tgt_classifier_conv2.load_state_dict(torch.load(tgt_classifiers[2]), strict=False)
tgt_encoder_conv2.load_state_dict(torch.load(tgt_encoders[2]), strict=False)
critic_conv2.load_state_dict(torch.load(critics[2]), strict=False)

tgt_classifier_conv3 = CielClassifier()
tgt_encoder_conv3 = CielEncoder()
critic_conv3 = Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims)
tgt_classifier_conv3.load_state_dict(torch.load(tgt_classifiers[3]), strict=False)
tgt_encoder_conv3.load_state_dict(torch.load(tgt_encoders[3]), strict=False)
critic_conv3.load_state_dict(torch.load(critics[3]), strict=False)

tgt_classifier_conv4 = DioneClassifier()
tgt_encoder_conv4 = DioneEncoder()
critic_conv4 = Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims)
tgt_classifier_conv4.load_state_dict(torch.load(tgt_classifiers[4]), strict=False)
tgt_encoder_conv4.load_state_dict(torch.load(tgt_encoders[4]), strict=False)
critic_conv4.load_state_dict(torch.load(critics[4]), strict=False)

tgt_classifier_convs = [tgt_classifier_conv1, tgt_classifier_conv2, tgt_classifier_conv3, tgt_classifier_conv4]
tgt_encoder_convs = [tgt_encoder_conv1, tgt_encoder_conv2, tgt_encoder_conv3, tgt_encoder_conv4]
critic_convs = [critic_conv1, critic_conv2, critic_conv3, critic_conv4]

for x in testing_xs:
    x = np.expand_dims(x, axis=0)

    for i in range(0, 4):
        encoded = tgt_encoder_convs[i](x)
        criticized = critic_convs[i](x)
        print(criticized)
        classified = tgt_classifier_convs[i](x)
