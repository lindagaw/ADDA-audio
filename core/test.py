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

def get_distribution(src_encoder, tgt_encoder, data_loader):

    vectors = []

    mahalanobis = []

    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()
        encoded_by_src = src_encoder(images).detach().cpu().numpy()
        encoded_by_tgt = tgt_encoder(images).detach().cpu().numpy()

        for val_encoded_by_src, val_encoded_by_tgt in zip(encoded_by_src, encoded_by_tgt):
            vector = np.linalg.norm(np.vstack((val_encoded_by_src, val_encoded_by_tgt)).tolist())
            vectors.append(vector)

    vectors = np.asarray(vectors)
    inv = np.cov(vectors)
    mean = np.mean(vectors)

    for vector in vectors:
        diff = vector - mean
        mahalanobis_dist = diff * inv * diff
        mahalanobis.append(mahalanobis_dist)

    mahalanobis = np.asarray(mahalanobis)
    mahalanobis_mean = np.mean(mahalanobis)
    mahalanobis_std = np.std(mahalanobis)

    return inv, mean, mahalanobis_mean, mahalanobis_std

def is_in_distribution(sample, inv, mean, mahalanobis_mean, mahalanobis_std):
    upper_coeff = 5000
    lower_coeff = 5000

    sample = sample.detach().cpu().numpy()

    m = np.linalg.norm((sample - mean) * inv * (sample - mean))

    if mahalanobis_mean - lower_coeff * mahalanobis_std < m and \
        m < mahalanobis_mean + upper_coeff * mahalanobis_std:
        return True
    else:
        return False


def eval_tgt_ood(src_encoder, tgt_encoder, src_classifier, tgt_classifier, src_data_loader, tgt_data_loader, data_loader):

    src_inv, src_mean, src_mahalanobis_mean, src_mahalanobis_std = \
                get_distribution(src_encoder, tgt_encoder, src_data_loader)

    tgt_inv, tgt_mean, tgt_mahalanobis_mean, tgt_mahalanobis_std = \
                get_distribution(src_encoder, tgt_encoder, tgt_data_loader)

    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    src_encoder.eval()
    tgt_encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0
    f1 = 0

    ys_pred = []
    ys_true = []

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()

        for image, label in zip(images, labels):
            #image = image.detach().cpu().numpy()
            label = label.detach().cpu().numpy()

            if not is_in_distribution(image, tgt_inv, tgt_mean, tgt_mahalanobis_mean, tgt_mahalanobis_std) and \
                not is_in_distribution(image, src_inv, src_mean, src_mahalanobis_mean, src_mahalanobis_std):
                continue
            elif is_in_distribution(image, tgt_inv, tgt_mean, tgt_mahalanobis_mean, tgt_mahalanobis_std):
                y_pred = np.argmax(np.squeeze(src_classifier(tgt_encoder(torch.unsqueeze(image, dim=0))).detach().cpu().numpy()))
            else:
                y_pred = np.argmax(np.squeeze(tgt_classifier(src_encoder(torch.unsqueeze(image, dim=0))).detach().cpu().numpy()))

            ys_pred.append(y_pred)
            ys_true.append(label)

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)
    #f1 /= len(data_loader.dataset)


    f1 = get_f1(ys_pred, ys_true, 'macro')
    f1_weighted = get_f1(ys_pred, ys_true, 'weighted')

    print("Avg Loss = {}, F1 = {:2%}, Weighted F1 = {:2%}".format(loss, f1, f1_weighted))
