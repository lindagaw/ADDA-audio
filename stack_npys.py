import os
import numpy as np
from sklearn.model_selection import train_test_split


neg_path = 'D://Datasets//Conflict//npy_neg//'
pos_path = 'D://Datasets//Conflict//npy_pos//'


def many_to_one(path, train_or_test, dest_dir, label):
    xs = []
    ys = []

    valid_npys = [path + file for file in os.listdir(path) if file.endswith('.npy')]

    for npy in valid_npys:
        xs.append(np.load(npy))
        if 'neg' in path and not 'pos' in path:
            ys.append([1, 0])
        elif 'pos' in path and not 'neg' in path:
            ys.append([0, 1])

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.33, random_state=42)

    print(xs.shape)
    print(ys.shape)

    if train_or_test == 'train':
        np.save(dest_dir + label+'_conflict_training_xs.npy', X_train)
        np.save(dest_dir + label+'_conflict_training_ys.npy', y_train)
    else:
        np.save(dest_dir + label+'_conflict_testing_xs.npy', X_test)
        np.save(dest_dir + label+'_conflict_testing_ys.npy', y_test)

path_neg = 'D://Datasets//Conflict//npy_neg//'
path_pos = 'D://Datasets//Conflict//npy_pos//'
conflict_npy_path = 'D://Datasets//CONFLICT//'
dest_dir = conflict_npy_path
for i in ['train', 'test']:
    for j in ['pos', 'neg']:
        many_to_one(path_neg, i, conflict_npy_path, j)
        many_to_one(path_pos, i, conflict_npy_path, j)

train_xs = np.vstack((np.load(dest_dir + 'pos_conflict_training_xs.npy'), np.load(dest_dir + 'neg_conflict_training_xs.npy')))
train_ys = np.vstack((np.load(dest_dir + 'pos_conflict_training_ys.npy'), np.load(dest_dir + 'neg_conflict_training_ys.npy')))

test_xs = np.vstack((np.load(dest_dir + 'pos_conflict_testing_xs.npy'), np.load(dest_dir + 'neg_conflict_testing_xs.npy')))
test_ys = np.vstack((np.load(dest_dir + 'pos_conflict_testing_ys.npy'), np.load(dest_dir + 'neg_conflict_testing_ys.npy')))

np.save(dest_dir + 'conflict_training_xs.npy', train_xs)
np.save(dest_dir + 'conflict_training_ys.npy', train_ys)

np.save(dest_dir + 'conflict_testing_xs.npy', test_xs)
np.save(dest_dir + 'conflict_testing_ys.npy', test_ys)
