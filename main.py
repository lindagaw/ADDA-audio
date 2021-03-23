"""Main script for ADDA."""

from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

import params_for_main as params
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, GalateaEncoder, GalateaClassifier

from models import AurielEncoder, BeatriceEncoder, CielEncoder, DioneEncoder
from models import AurielClassifier, BeatriceClassifier, CielClassifier, DioneClassifier
from utils import get_data_loader, init_model, init_random_seed

from source_classifier import load_chopped_source_model, load_second_half_chopped_source_model

def group(conv):

    d_model_restore = "snapshots//CONV_" + str(conv) + "_ACTIVATIONS-ADDA-critic-final.pt"
    src_classifier_restore = "snapshots//CONV_" + str(conv) + "_ACTIVATIONS-ADDA-source-classifier-final.pt"
    tgt_encoder_restore = "snapshots//CONV_" + str(conv) + "_ACTIVATIONS-ADDA-target-encoder-final.pt"
    src_encoder_restore = "snapshots//CONV_" + str(conv) + "_ACTIVATIONS-ADDA-source-encoder-final.pt"
    if int(conv) == 1:
        src_encoder = init_model(net=AurielEncoder(),
                                 restore=src_encoder_restore)
        src_classifier = init_model(net=AurielClassifier(),
                                    restore=src_classifier_restore)
        tgt_encoder = init_model(net=AurielEncoder(),
                                 restore=tgt_encoder_restore)
    elif int(conv) == 2:
        src_encoder = init_model(net=BeatriceEncoder(),
                                 restore=src_encoder_restore)
        src_classifier = init_model(net=BeatriceClassifier(),
                                    restore=src_classifier_restore)
        tgt_encoder = init_model(net=BeatriceEncoder(),
                                 restore=tgt_encoder_restore)
    elif int(conv) == 3:
        src_encoder = init_model(net=CielEncoder(),
                                 restore=src_encoder_restore)
        src_classifier = init_model(net=CielClassifier(),
                                    restore=src_classifier_restore)
        tgt_encoder = init_model(net=CielEncoder(),
                                 restore=tgt_encoder_restore)
    elif int(conv) == 4:
        src_encoder = init_model(net=DioneEncoder(),
                                 restore=src_encoder_restore)
        src_classifier = init_model(net=DioneClassifier(),
                                    restore=src_classifier_restore)
        tgt_encoder = init_model(net=DioneEncoder(),
                                 restore=tgt_encoder_restore)
    else:
        raise RuntimeError("conv number must be 1, 2, 3, or 4.")

    critic = init_model(Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims),
                        restore=d_model_restore)

    return src_encoder, src_classifier, tgt_encoder, critic


if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)
    '''
    the purpose of this file is to test the enforced transfer + ADDA.
    dataset needed will be the testing set from CONFLICT.

    source encoder, source classifier, and target encoder should be pretrained.
    '''

    tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False, dataset='conflict')
