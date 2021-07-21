"""Main script for ADDA."""

from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

import sound_params as params
from core import eval_src, eval_tgt, train_src, train_tgt, eval_tgt_ood, train_tgt_classifier
from models import Discriminator, GalateaEncoder, GalateaClassifier

from models import AurielEncoder, BeatriceEncoder, CielEncoder, DioneEncoder
from models import AurielClassifier, BeatriceClassifier, CielClassifier, DioneClassifier
from utils import get_data_loader, init_model, init_random_seed

import sys

def trio(tgt_classifier_net, tgt_encoder_net, src_dataset, tgt_dataset, conv):
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
    '''
    tgt_classifier_2, tgt_encoder_2, critic_2 = \
        trio(BeatriceClassifier(), BeatriceEncoder(), 'CONV_2_ACTIVATIONS', 'CONV_2_ACTIVATIONS', 2)
    tgt_classifier_3, tgt_encoder_3, critic_3 = \
        trio(CielClassifier(), CielEncoder(), 'CONV_3_ACTIVATIONS', 'CONV_3_ACTIVATIONS', 3)
    tgt_classifier_4, tgt_encoder_4, critic_4 = \
        trio(DioneClassifier(), DioneEncoder(), 'CONV_4_ACTIVATIONS', 'CONV_4_ACTIVATIONS', 4)
    '''