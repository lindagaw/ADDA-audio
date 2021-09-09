"""Test script to classify target data."""

import torch
import torch.nn as nn

from utils import make_variable
from utils import get_f1

import os
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import f1_score

def eval_tgt_with_probe(encoder, critic, src_classifier, tgt_classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    src_classifier.eval()
    tgt_classifier.eval()
    # init loss and accuracy
    loss = 0
    acc = 0
    f1 = 0

    ys_pred = []
    ys_true = []
    # set loss function
    criterion = nn.CrossEntropyLoss()
    flag = False
    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()

        probeds = critic(encoder(images))

        for image, label, probed in zip(images, labels, probeds):
            if torch.argmax(probed) == 1:
                pred = torch.argmax(src_classifier(encoder(torch.unsqueeze(image, 0)))).detach().cpu().numpy()
            else:
                pred = torch.argmax(tgt_classifier(encoder(torch.unsqueeze(image, 0)))).detach().cpu().numpy()

        ys_pred.append(np.squeeze(pred))
        ys_true.append(np.squeeze(label.detach().cpu().numpy()))  

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)
    #f1 /= len(data_loader.dataset)
    f1 = get_f1(ys_pred, ys_true, 'macro')
    f1_weighted = get_f1(ys_pred, ys_true, 'weighted')

    print("Avg Loss = {}, F1 = {:2%}, Weighted F1 = {:2%}".format(loss, f1, f1_weighted))


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

def get_distribution(src_encoder, tgt_encoder, src_classifier, tgt_classifier, critic, data_loader, which_data_loader):

    print("Start calculating the mahalanobis distances' mean and standard deviation ... ")
    vectors = []
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True).squeeze_()
        labels = make_variable(labels).squeeze_()
        torch.no_grad()
        src_preds = src_classifier(torch.squeeze(src_encoder(images))).detach().cpu().numpy()
        tgt_preds = tgt_classifier(torch.squeeze(tgt_encoder(images))).detach().cpu().numpy()
        critic_at_src = critic(torch.squeeze(src_encoder(images))).detach().cpu().numpy()
        critic_at_tgt = critic(torch.squeeze(tgt_encoder(images))).detach().cpu().numpy()
        for image, label, src_pred, tgt_pred, src_critic, tgt_critic \
                        in zip(images, labels, src_preds, tgt_preds, critic_at_src, critic_at_tgt):
            vectors.append(np.linalg.norm(src_critic.tolist() + tgt_critic.tolist()))
            #print('processing vector ' + str(src_critic.tolist() + tgt_critic.tolist()))

        mean = np.asarray(vectors).mean(axis=0)
        cov = np.cov(vectors)
        try:
            iv = np.linalg.inv(cov)
        except:
            iv = cov
        mahalanobis = np.asarray([distance.mahalanobis(v, mean, iv) for v in vectors])
        mahalanobis_mean = np.mean(mahalanobis)
        mahalanobis_std = np.std(mahalanobis)
        np.save('snapshots//' + which_data_loader + '_mahalanobis_mean.npy', mahalanobis_mean)
        np.save('snapshots//' + which_data_loader + '_mahalanobis_std.npy', mahalanobis_std)
        np.save('snapshots//' + which_data_loader + '_iv.npy', iv)
        np.save('snapshots//' + which_data_loader + '_mean.npy', mean)

    print("Finished obtaining the mahalanobis distances' mean and standard deviation on " + which_data_loader)
    return mahalanobis_mean, mahalanobis_std, iv, mean

def is_in_distribution(vector, mahalanobis_mean, mahalanobis_std, mean, iv):
    upper_coefficient = 0.1
    lower_coefficient = 0.1

    upper = mahalanobis_mean + upper_coefficient * mahalanobis_std
    lower = mahalanobis_mean - lower_coefficient * mahalanobis_std

    mahalanobis = distance.mahalanobis(vector, mean, iv)

    if lower < mahalanobis and mahalanobis < upper:
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
    src_classifier.eval()
    tgt_classifier.eval()

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


def eval_ADDA(src_encoder, tgt_encoder, src_classifier, tgt_classifier, critic, data_loader):

    src_mahalanobis_std = np.load('snapshots//' + 'src' + '_mahalanobis_std.npy')
    src_mahalanobis_mean = np.load('snapshots//' + 'src' + '_mahalanobis_mean.npy')
    src_iv = np.load('snapshots//' + 'src' + '_iv.npy')
    src_mean = np.load('snapshots//' + 'src' + '_mean.npy')

    tgt_mahalanobis_std = np.load('snapshots//' + 'tgt' + '_mahalanobis_std.npy')
    tgt_mahalanobis_mean = np.load('snapshots//' + 'tgt' + '_mahalanobis_mean.npy')
    tgt_iv = np.load('snapshots//' + 'tgt' + '_iv.npy')
    tgt_mean = np.load('snapshots//' + 'tgt' + '_mean.npy')

    """Evaluation for target encoder by source classifier on target dataset."""
    tgt_encoder.eval()
    src_encoder.eval()
    src_classifier.eval()
    tgt_classifier.eval()
    # init loss and accuracy
    # set loss function
    criterion = nn.CrossEntropyLoss()
    # evaluate network

    y_trues = []
    y_preds = []

    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()
        torch.no_grad()

        src_preds = src_classifier(torch.squeeze(src_encoder(images))).detach().cpu().numpy()
        tgt_preds = tgt_classifier(torch.squeeze(tgt_encoder(images))).detach().cpu().numpy()
        critic_at_src = critic(torch.squeeze(src_encoder(images))).detach().cpu().numpy()
        critic_at_tgt = critic(torch.squeeze(tgt_encoder(images))).detach().cpu().numpy()

        for image, label, src_pred, tgt_pred, src_critic, tgt_critic \
                        in zip(images, labels, src_preds, tgt_preds, critic_at_src, critic_at_tgt):

            vector = np.linalg.norm(src_critic.tolist() + tgt_critic.tolist())

            # ouf of distribution:
            if not is_in_distribution(vector, tgt_mahalanobis_mean, tgt_mahalanobis_std, tgt_mean, tgt_iv) \
                and not is_in_distribution(vector, src_mahalanobis_mean, src_mahalanobis_std, src_mean, src_iv):
                continue
            # if in distribution which the target:
            elif is_in_distribution(vector, tgt_mahalanobis_mean, tgt_mahalanobis_std, tgt_mean, tgt_iv):
                y_pred = np.argmax(tgt_pred)
            else:
                y_pred = np.argmax(src_pred)

            #y_pred = np.argmax(tgt_pred)
            y_preds.append(y_pred)
            y_trues.append(label.detach().cpu().numpy())


    print("Avg F1 = {:2%}".format(f1_score(y_true=y_trues, y_pred=y_preds, average='weighted')))
