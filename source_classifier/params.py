first_half = '..//..//chopped_models//second_half//'
second_half = '..//..//chopped_models//second_half//'

data_root = "data"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 50
image_size = 64

# params for training network
num_gpu = 1
num_epochs_pre = 5
log_step_pre = 20
eval_step_pre = 20
save_step_pre = 100
num_epochs = 5
log_step = 100
save_step = 100
manual_seed = None

# params for optimizing models
d_learning_rate = 1e-4
c_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9
