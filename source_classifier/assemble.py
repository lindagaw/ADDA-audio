# assemble the keras model w/ the results of probes and enforcer_preds

import numpy as np
import torc

import load_chopped_source_model, load_second_half_chopped_source_model

xs_test = np.load('data//Conflict//' + 'conflict_testing_xs.npy')
ys_test = np.load('data//Conflict//' + 'conflict_testing_ys.npy')

# load conv 1:
conv_1 = load_chopped_source_model(1)
