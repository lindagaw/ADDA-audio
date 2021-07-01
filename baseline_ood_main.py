"""Main script for ADDA."""

from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

import params_for_main as params
from models import Essentia
from utils import get_data_loader, init_model, init_random_seed
from ood_baseline import eval, train, eval_ood

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    #activations = ['CONV_1_ACTIVATIONS', 'CONV_2_ACTIVATIONS', 'CONV_3_ACTIVATIONS', 'CONV_4_ACTIVATIONS']

    try:
        params.src_dataset = 'CONV_0_ACTIVATIONS'
        params.tgt_dataset = 'CONV_0_ACTIVATIONS'
    except:
        raise RuntimeError('must specify src and trg names in the form of \
                            python enforced_tf_main.py src_name_string, tgt_name_string.')

    print('src_dataset is the ' + params.src_dataset + ' of samples in emotion dataset (source).')
    print('tgt_dataset is the ' + params.tgt_dataset + ' of samples in conflict dataset (target).')
    # load dataset
    src_data_loader = get_data_loader(params.src_dataset, dataset='emotion')
    src_data_loader_eval = get_data_loader(params.src_dataset, train=False, dataset='emotion')
    tgt_data_loader = get_data_loader(params.tgt_dataset, dataset='conflict')
    tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False, dataset='conflict')


    src_classifier = init_model(net=Essentia(),
                                restore=params.src_classifier_restore)

    tgt_classifier = init_model(net=Essentia(),
                                restore=params.src_classifier_restore)


    # train source model
    print("=== Training classifier for source domain ===")
    print(">>> Source Classifier <<<")
    print(src_classifier)
    src_classifier = train(src_classifier, src_data_loader)
    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval(src_classifier, src_data_loader_eval)

    # train target model
    print("=== Training classifier for target domain ===")
    print(">>> Target Classifier <<<")
    print(tgt_classifier)
    tgt_classifier = train(tgt_classifier, tgt_data_loader)
    # eval target model
    print("=== Evaluating classifier for target domain ===")
    eval(tgt_classifier, tgt_data_loader_eval)

    print("=== Evaluating src classifier for target domain without OOD ===")
    eval(src_classifier, tgt_data_loader_eval)

    print("=== Evaluating src classifier for target domain with OOD ===")
    eval_ood(src_classifier, src_data_loader, tgt_data_loader_eval)