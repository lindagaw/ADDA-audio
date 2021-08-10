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
ys_train_src = from_one_hot(np.load(emotion_path+'emotion_training_ys.npy'))
ys_test_src = from_one_hot(np.load(emotion_path+'emotion_testing_ys.npy'))

xs_train_tgt = np.load(conflict_path+'conflict_training_xs.npy')
xs_test_tgt = np.load(conflict_path+'conflict_testing_xs.npy')
ys_train_tgt = from_one_hot(np.load(conflict_path+'conflict_training_ys.npy'))
ys_test_tgt = from_one_hot(np.load(conflict_path+'conflict_testing_ys.npy'))

# calculate the empirical mean of xs_train
def get_distribution(xs):
    vectors = []
    mahalanobis = []
    for x in xs:
        norm = np.linalg.norm(x)
        vectors.append(norm)
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

inv, mean, mahalanobis_mean, mahalanobis_std = get_distribution(xs_train_src)

def is_in_distribution(sample, inv, mean, mahalanobis_mean, mahalanobis_std):
    upper_coeff = 85000
    lower_coeff = 85000

    m = np.linalg.norm((sample - mean) * inv * (sample - mean))

    print(m-mahalanobis_mean)
    print(mahalanobis_std)
    print('--------------------')

    if mahalanobis_mean - lower_coeff * mahalanobis_std < m and \
        m < mahalanobis_mean + upper_coeff * mahalanobis_std:
        return True
    else:
        return False

# calculate f1 with OOD
y_pred = []
y_true = []

for sample, true_label in zip(xs_test_tgt, ys_test_tgt):

    if is_in_distribution(sample, inv, mean, mahalanobis_mean, mahalanobis_std):
        pred = np.argmax(model.predict(np.asarray([sample])))
        y_pred.append(pred)
        y_true.append(true_label)

print(y_true)
print(y_pred)
print(f1_score(y_true, y_pred, average='weighted'))
