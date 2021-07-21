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


tgt_classifier_conv0 = GalateaClassifier()
tgt_encoder_conv0 = GalateaEncoder()
critic_conv0 = Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims)
tgt_classifier_conv0.load_state_dict(tgt_classifiers[0])
tgt_encoder_conv0.load_state_dict(tgt_encoder[0])
critic_conv0.load_state_dict(critics[0])

tgt_classifier_conv1 = AurielClassifier()
tgt_encoder_conv1 = AurielEncoder()
critic_conv1 = Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims)
tgt_classifier_conv1.load_state_dict(tgt_classifiers[1])
tgt_encoder_conv1.load_state_dict(tgt_encoder[1])
critic_conv1.load_state_dict(critics[1])

tgt_classifier_conv2 = BeatriceClassifier()
tgt_encoder_conv2 = BeatriceEncoder()
critic_conv2 = Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims)
tgt_classifier_conv2.load_state_dict(tgt_classifiers[2])
tgt_encoder_conv2.load_state_dict(tgt_encoder[2])
critic_conv2.load_state_dict(critics[2])

tgt_classifier_conv3 = CielClassifier()
tgt_encoder_conv3 = CielEncoder()
critic_conv3 = Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims)
tgt_classifier_conv3.load_state_dict(tgt_classifiers[3])
tgt_encoder_conv3.load_state_dict(tgt_encoder[3])
critic_conv3.load_state_dict(critics[3])

tgt_classifier_conv4 = DioneClassifier()
tgt_encoder_conv4 = DioneEncoder()
critic_conv4 = Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims)
tgt_classifier_conv4.load_state_dict(tgt_classifiers[4])
tgt_encoder_conv4.load_state_dict(tgt_encoder[4])
critic_conv4.load_state_dict(critics[4])

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
