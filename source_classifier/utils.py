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
