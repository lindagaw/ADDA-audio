# assemble the keras model w/ the results of probes and enforcer_preds

import numpy as np
import torch
import params
from load_1st_half_chopped_source_model import load_chopped_source_model
from load_2nd_half_chopped_source_model import load_second_half_chopped_source_model
from utils import init_random_seed
xs_test = np.load('data//Conflict//' + 'conflict_testing_xs.npy')
ys_test = np.load('data//Conflict//' + 'conflict_testing_ys.npy')

import sys
import os
from tensorflow import keras
from keras.models import load_model
from keras.models import Sequential

model = load_model('..//model.hdf5')

'''
for a given input x, if probe_1(x) == 1 ==> exit at enforcer_1(x)
elif: probe_2(x) == 1 ==> exit at enforcer_2(x)
elif: probe_3(x) == 1 ==> exit at enforcer_3(x)
elif: probe_4(x) == 1 ==> exit at enforcer_4(x)
else: exit at rest_of_src_classifier(x)
'''
init_random_seed(params.manual_seed)

# index 0 --> conv 1
feats_after_enforcers = [np.squeeze(np.asarray(torch.load(path).cpu())) for path in params.feats_after_enforcers]
preds_after_enforcers = [np.squeeze(np.asarray(torch.load(path).cpu())) for path in params.preds_after_enforcers]
preds_after_probes = [np.squeeze(np.asarray(torch.load(path).cpu())) for path in params.preds_after_probes]

ys_pred = []

for index in range(0, len(xs_test)):
    x = xs_test[index]
    print('processing ' + str(index) + ' th element ...')
    for conv in range(0, 4):
        probe = preds_after_probes[conv][index]
        if probe == 1:
            y_pred = preds_after_enforcers[conv][index]
            ys_pred.append(y_pred)
            break

    y_pred = np.squeeze(model.predict(np.expand_dims(x, axis=0)))
    ys_pred.append(y_pred)

ys_pred = np.asarray(ys_pred)
print(ys_pred.shape)
#print(xs_test.shape)
#print(ys_test.shape)
#print(feats_after_enforcers[0].shape)
#print(preds_after_enforcers[0].shape)
#print(preds_after_probes[0].shape)
