"""Utilities for ADDA."""

import os
import random
from sklearn.metrics import f1_score
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import sound_params as params
from datasets import get_mnist, get_usps, get_emotion, get_conflict
from datasets import get_conv_0_activations, get_conv_1_activations
from datasets import get_conv_2_activations, get_conv_3_activations
from datasets import get_conv_4_activations

from models import AurielEncoder, BeatriceEncoder, CielEncoder, DioneEncoder
from models import AurielClassifier, BeatriceClassifier, CielClassifier, DioneClassifier

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


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_loader(name, train=True, dataset=None):
    """Get data loader by name."""
    if name == "MNIST":
        return get_mnist(train)
    elif name == "USPS":
        return get_usps(train)
    elif name == "EMOTION":
        return get_emotion(train)
    elif name == "CONFLICT":
        return get_conflict(train)
    elif name == 'CONV_0_ACTIVATIONS':
        return get_conv_0_activations(train, dataset=dataset)
    elif name == 'CONV_1_ACTIVATIONS':
        return get_conv_1_activations(train, dataset=dataset)
    elif name == 'CONV_2_ACTIVATIONS':
        return get_conv_2_activations(train, dataset=dataset)
    elif name == 'CONV_3_ACTIVATIONS':
        return get_conv_3_activations(train, dataset=dataset)
    elif name == 'CONV_4_ACTIVATIONS':
        return get_conv_4_activations(train, dataset=dataset)


def init_model(net, restore):
    """Init models with cuda and weights."""
    # init weights of model
    net.apply(init_weights)

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net


def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    torch.save(net.state_dict(),
               os.path.join(params.model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(params.model_root,
                                                             filename)))


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
