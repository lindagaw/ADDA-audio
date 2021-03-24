# assemble the keras model w/ the results of probes and enforcer_preds

import numpy as np
import torch
import params
from load_1st_half_chopped_source_model import load_chopped_source_model
from load_2nd_half_chopped_source_model import load_second_half_chopped_source_model
from utils import init_random_seed
xs_test = np.load('data//Conflict//' + 'conflict_testing_xs.npy')
ys_test = np.load('data//Conflict//' + 'conflict_testing_ys.npy')

'''
for a given input x, if probe_1(x) == 1 ==> exit at enforcer_1(x)
elif: probe_2(x) == 1 ==> exit at enforcer_2(x)
elif: probe_3(x) == 1 ==> exit at enforcer_3(x)
elif: probe_4(x) == 1 ==> exit at enforcer_4(x)
else: exit at rest_of_src_classifier(x)
'''
init_random_seed(params.manual_seed)

# index 0 --> conv 1
feats_after_enforcers = [torch.load(path) for path in params.feats_after_enforcers]
preds_after_enforcers = [torch.load(path) for path in params.preds_after_enforcers]
preds_after_probes = [torch.load(path) for path in params.preds_after_probes]

print(xs.shape)
print(feats_after_enforcers[0].shape)
print(preds_after_enforcers[0].shape)
print(preds_after_probes[0].shape)
