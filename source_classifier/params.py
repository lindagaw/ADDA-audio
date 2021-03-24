first_half = '..//chopped_models//second_half//'
second_half = '..//chopped_models//second_half//'

all_feats_by_conv_1_after_enforcer = 'result//all_feats_by_conv_1_after_enforcer.pt'
all_feats_by_conv_2_after_enforcer = 'result//all_feats_by_conv_2_after_enforcer.pt'
all_feats_by_conv_3_after_enforcer = 'result//all_feats_by_conv_3_after_enforcer.pt'
all_feats_by_conv_4_after_enforcer = 'result//all_feats_by_conv_4_after_enforcer.pt'
feats_after_enforcers = [all_feats_by_conv_1_after_enforcer, all_feats_by_conv_2_after_enforcer, \
                        all_feats_by_conv_3_after_enforcer, all_feats_by_conv_4_after_enforcer]

all_preds_by_conv_1_after_enforcer = 'result//all_preds_by_conv_1_after_enforcer.pt'
all_preds_by_conv_2_after_enforcer = 'result//all_preds_by_conv_2_after_enforcer.pt'
all_preds_by_conv_3_after_enforcer = 'result//all_preds_by_conv_3_after_enforcer.pt'
all_preds_by_conv_4_after_enforcer = 'result//all_preds_by_conv_4_after_enforcer.pt'
preds_after_enforcers = [all_preds_by_conv_1_after_enforcer, all_preds_by_conv_2_after_enforcer, \
                        all_preds_by_conv_3_after_enforcer, all_preds_by_conv_4_after_enforcer]

all_preds_by_conv_1_after_probe = 'result//all_preds_by_conv_1_after_probe.pt'
all_preds_by_conv_2_after_probe = 'result//all_preds_by_conv_2_after_probe.pt'
all_preds_by_conv_3_after_probe = 'result//all_preds_by_conv_3_after_probe.pt'
all_preds_by_conv_4_after_probe = 'result//all_preds_by_conv_4_after_probe.pt'
preds_after_probes = [all_preds_by_conv_1_after_probe, all_preds_by_conv_2_after_probe, \
                        all_preds_by_conv_3_after_probe, all_preds_by_conv_3_after_probe]

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
