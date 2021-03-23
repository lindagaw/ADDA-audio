import os
import random
from sklearn.metrics import f1_score
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

def get_f1(ys_pred, ys_true, average):
    return f1_score(ys_true.cpu(), ys_pred.cpu(), average=average)

def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)

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
