import os
from tensorflow import keras
from keras.models import load_model
import params
def load_chopped_source_model(conv):
    path = params.first_half
    models = sorted(os.listdir(path))

    #conv = 4
    index = int(conv) - 1

    model = load_model(path + models[index])
    print('model loaded at ' + path + models[index])
    model.summary()

    return model
