# assemble the keras model w/ the results of probes and enforcer_preds

import numpy as np
import torc
import params
import load_chopped_source_model, load_second_half_chopped_source_model

#xs_test = np.load('data//Conflict//' + 'conflict_testing_xs.npy')
#ys_test = np.load('data//Conflict//' + 'conflict_testing_ys.npy')

'''
for a given input x, if probe_1(x) == 1 ==> exit at enforcer_1(x)
elif: probe_2(x) == 1 ==> exit at enforcer_2(x)
elif: probe_3(x) == 1 ==> exit at enforcer_3(x)
elif: probe_4(x) == 1 ==> exit at enforcer_4(x)
else: exit at rest_of_src_classifier(x)
'''
feats_after_enforcers = params.feats_after_enforcers
preds_after_enforcers = params.preds_after_enforcers
preds_after_probes = params.preds_after_probes

s = load_second_half_chopped_source_model(conv=4)
print(s.shape)
