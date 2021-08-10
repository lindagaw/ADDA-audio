import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import f1_score

model = tf.keras.models.load_model('..//..//model.hdf5')

# convert shape (,2) to (,5)
def from_one_hot(ys):
    returned = []
    for y in ys:
        returned.append(np.argmax(y))

    return np.asarray(returned)


emotion_path = '..//..//..//Datasets//EMOTION//'
conflict_path = '..//..//..//Datasets//CONFLICT//'

xs_train_src = np.load(emotion_path+'emotion_training_xs.npy')
xs_test_src = np.load(emotion_path+'emotion_testing_xs.npy')
ys_train_src = np.load(emotion_path+'emotion_training_ys.npy')
ys_test_src = np.load(emotion_path+'emotion_testing_ys.npy')

xs_train_tgt = np.load(conflict_path+'conflict_training_xs.npy')
xs_test_tgt = np.load(conflict_path+'conflict_testing_xs.npy')
ys_train_tgt = from_one_hot(np.load(conflict_path+'conflict_training_ys.npy'))
ys_test_tgt = from_one_hot(np.load(conflict_path+'conflict_testing_ys.npy'))

# calculate the empirical mean of xs_train
ys_predicted = from_one_hot(model.predict(xs_test_tgt))
print(ys_predicted)
print(f1_score(ys_test_tgt, ys_predicted, average='weighted'))
