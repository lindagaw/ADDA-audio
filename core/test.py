"""Test script to classify target data."""

import torch
import torch.nn as nn

from utils import make_variable
from utils import get_f1

import os
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import f1_score




def eval_tgt(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0
    f1 = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()
    flag = False
    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()

        preds = classifier(encoder(images))
        loss += criterion(preds, labels).data # criterion is cross entropy loss

        pred_cls = preds.data.max(1)[1]
        #f1 += get_f1(pred_cls, labels.data, average='macro')
        acc += pred_cls.eq(labels.data).cpu().sum()

        if not flag:
            ys_pred = pred_cls
            ys_true = labels
            flag = True
        else:
            ys_pred = torch.cat((ys_pred, pred_cls), 0)
            ys_true = torch.cat((ys_true, labels), 0)

    loss = loss.float()
    acc = acc.float()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)
    #f1 /= len(data_loader.dataset)
    f1 = get_f1(ys_pred, ys_true, 'macro')
    f1_weighted = get_f1(ys_pred, ys_true, 'weighted')

    print("Avg Loss = {}, F1 = {:2%}, Weighted F1 = {:2%}".format(loss, f1, f1_weighted))
