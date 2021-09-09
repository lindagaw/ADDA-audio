"""Pre-train encoder and classifier for source dataset."""
import torch
import torch.nn as nn
import torch.optim as optim

import sound_params as params
from utils import make_variable, save_model, get_f1

import tensorflow as tf

import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import accuracy_score

def train(classifier, data_loader):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    classifier.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(classifier.parameters()),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2))
    #criterion = nn.CrossEntropyLoss(weights=tf.convert_to_tensor([1, 300]))
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_pre):

        for step, (images, labels) in enumerate(data_loader):
            # make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = classifier(images)
            loss = criterion(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len(data_loader),
                              loss.data))

        # eval model on test set
        if ((epoch + 1) % params.eval_step_pre == 0):
            eval(classifier, data_loader)

    return classifier


def eval(classifier, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
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
        labels = make_variable(labels)

        preds = classifier(images)
        loss += criterion(preds, labels).data

        pred_cls = preds.data.max(1)[1]
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
    f1 = get_f1(ys_pred, ys_true, 'weighted')
    #f1_weighted = get_f1(ys_pred, ys_true, 'weighted')

    print("Avg Loss = {}, F1 = {:2%}, accuracy = {:2%}".format(loss, f1, acc))

def get_distribution(data_loader):

    vectors = []

    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True).detach().cpu().numpy()
        labels = make_variable(labels).detach().cpu().numpy()

        vectors.append(np.linalg.norm(images))

    mean = np.mean(vectors)
    inv = np.cov(vectors)
    m_dists = [ (x-mean) * inv * (x-mean) for x in vectors ]

    m_mean = np.mean(m_dists)
    m_std = np.std(m_dists)

    return mean, inv, m_mean, m_std

def is_in_distribution(sample, mean, inv, m_mean, m_std):
    upper_coeff = 1500
    lower_coeff = 1500

    upper = m_mean + upper_coeff*m_std
    lower = m_mean - lower_coeff*m_std

    m = np.linalg.norm((sample-mean)*inv*(sample-mean))

    if lower < m and m < upper:
        return True
    else:
        return False

def eval_ood(src_classifier, src_data_loader, tgt_data_loader_eval):
    mean, inv, m_mean, m_std = get_distribution(src_data_loader)

    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    src_classifier.eval()

    acc = 0
    f1 = 0


    ys_true = []
    ys_pred = []

    for (images, labels) in tgt_data_loader_eval:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).detach().cpu().numpy()

        preds = src_classifier(images)
        
        for pred, image, label in zip(preds, images, labels):
            if is_in_distribution(image.detach().cpu().numpy(), mean, inv, m_mean, m_std):
                ys_true.append(label)
                ys_pred.append(np.argmax(pred.detach().cpu().numpy()))
            else:
                continue
    acc = accuracy_score(ys_true, ys_pred)
    f1 = get_f1(ys_pred, ys_true, 'macro')


    print(" F1 = {:2%}, accuracy = {:2%}".format(f1, acc))