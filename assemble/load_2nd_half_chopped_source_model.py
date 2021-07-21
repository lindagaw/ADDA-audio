import os
import sys
import os
from tensorflow import keras
from keras.models import load_model
from keras.models import Sequential
from tensorflow.keras.layers import Add, Input
import params

def load_second_half_chopped_source_model(conv):
    path = params.second_half
    models = sorted(os.listdir(path))

    #conv = 4
    index = int(conv) - 1

    model = load_model(path + models[index])
    print('model laoded at ' + path + models[index])
    model.summary()

    return model
