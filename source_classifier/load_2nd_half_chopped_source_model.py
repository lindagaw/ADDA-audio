import os
import sys
import os
from tensorflow import keras
from keras.models import load_model
from keras.models import Sequential
from tensorflow.keras.layers import Add, Input
import params
conv = int(sys.argv[1])
n = 3 * conv

def load_second_half_chopped_source_model(conv):
    path = params.second_half
    models = sorted(os.listdir(path))

    #conv = 4
    index = int(conv) - 1

    model = load_model(path + models[index])
    print('model laoded at ' + path + models[index])
    model.summary()

    return model

'''
model_folder = '..//module//five_class_ood//'

def list_source_models(model_folder):
    usable_models = [model for model in os.listdir(model_folder) if '.hdf5' in model]
    return usable_models

def obtain_source_model():
    usable_models = list_source_models(model_folder)
    model = load_model(model_folder + usable_models[0])
    return model

model = obtain_source_model()
print('finished loaing the source model.')

salvaged = Sequential()

for layer_index in range(0, len(model.layers)):
    if layer_index == n - 1:
        output_shape_from_the_top_half = model.layers[layer_index].output_shape
        #input_layer = Input(shape=output_shape_from_the_top_half)

    elif layer_index >= n:
        salvaged.add(model.layers[layer_index])

print('input shape = ' + str(output_shape_from_the_top_half))

salvaged.build(input_shape=output_shape_from_the_top_half)
salvaged.summary()

dest = 'chopped_models//second_half//'
hdf5 = dest + 'bottom_' + str(conv) + '_conv_layers_preserved.hdf5'

if not os.path.isdir(dest):
    os.makedirs(dest)

salvaged.save(hdf5)
print('second half chopped source model saved at ' + hdf5)
'''
