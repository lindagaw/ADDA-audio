# assemble the keras model w/ the results of probes and enforcer_preds

import numpy as np
import torch
import params
#from load_1st_half_chopped_source_model import load_chopped_source_model
#from load_2nd_half_chopped_source_model import load_second_half_chopped_source_model
from utils import init_random_seed
xs_test = np.load('data//Conflict//' + 'conflict_testing_xs.npy')
ys_test = [list(r).index(1) for r in np.load('data//Conflict//' + 'conflict_testing_ys.npy')]

import sys
import os
from tensorflow import keras
from keras.models import load_model
from keras.models import Sequential
from sklearn.metrics import f1_score


print('loading the source classifier ...')
model = load_model('..//model.hdf5')
print('finished loading the source classifier ...')
init_random_seed(params.manual_seed)

# index 0 --> conv 1
feats_after_enforcers = [np.squeeze(np.asarray(torch.load(path).cpu())) for path in params.feats_after_enforcers]
preds_after_enforcers = [np.squeeze(np.asarray(torch.load(path).cpu())) for path in params.preds_after_enforcers]
preds_after_probes = [np.squeeze(np.asarray(torch.load(path).cpu())) for path in params.preds_after_probes]

ys_pred = []
ys_true = []

for index in range(0, len(xs_test)):
    try:
        x = xs_test[index]
        y_true = ys_test[index]
        #print('processing ' + str(index) + ' th element ...')

        flag = False
        for conv in range(0, 4):
            probe = preds_after_probes[conv][index]
            if probe == 1:
                y_pred = preds_after_enforcers[conv][index]
                ys_pred.append(y_pred)
                ys_true.append(y_true)
                flag = True
                break
        if flag:
            continue
        else:
            y_pred = np.squeeze(model.predict(np.expand_dims(x, axis=0)))
            ys_pred.append(y_pred)
            ys_true.append(y_true)
            flag = False

    except Exception as e:
        print(e)

ys_pred = np.asarray([np.argmax(val) for val in ys_pred])
ys_true = np.asarray(ys_true)


np.save('ys_pred.npy', ys_pred)
np.save('ys_true.npy', ys_true)

print(ys_pred.shape)
print(ys_true.shape)
f1 = f1_score(ys_true, ys_pred, average='weighted')
print(f1)
#print(xs_test.shape)
#print(ys_test.shape)
#print(feats_after_enforcers[0].shape)
#print(preds_after_enforcers[0].shape)
#print(preds_after_probes[0].shape)
