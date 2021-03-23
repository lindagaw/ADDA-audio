import os
from tensorflow import keras
from keras.models import load_model

def load_chopped_source_model(conv):
    path = 'D://GitHub//Conflict_Detection//chopped_models//first_half//'
    models = sorted(os.listdir(path))

    #conv = 4
    index = conv - 1

    model = load_model(path + models[index])
    print('model laoded at ' + path + models[index])
    model.summary()

    return model
