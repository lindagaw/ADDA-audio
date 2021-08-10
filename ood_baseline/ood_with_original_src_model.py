import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.load_model('..//..//model.hdf5')

# convert shape (,2) to (,5)
def conflict_to_anger(ys):
    returned = []
    for y in ys:
        if np.argmax(y) == 0:
            returned.append(np.asarray([1, 0, 0, 0, 0]))
        else:
            returned.append(np.asarray([0, 1, 0, 0, 0]))

    return np.asarray(returned)


emotion_path = '..//..//..//Datasets//EMOTION//'
conflict_path = '..//..//..//Datasets//CONFLICT//'

xs_train_src = np.load(emotion_path+'emotion_training_xs.npy')
xs_test_src = np.load(emotion_path+'emotion_testing_xs.npy')
ys_train_src = np.load(emotion_path+'emotion_training_ys.npy')
ys_test_src = np.load(emotion_path+'emotion_testing_ys.npy')

xs_train_tgt = np.load(conflict_path+'conflict_training_xs.npy')
xs_test_tgt = np.load(conflict_path+'conflict_testing_xs.npy')
ys_train_tgt = conflict_to_anger(np.load(conflict_path+'conflict_training_ys.npy'))
ys_test_tgt = conflict_to_anger(np.load(conflict_path+'conflict_testing_ys.npy'))

# calculate the empirical mean of xs_train
