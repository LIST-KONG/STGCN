import graphcnn.setup.fmri_pre_process as fmri_pre_proc
from graphcnn.experiment import *
import numpy as np
import scipy.io


class GCLSTMHiNetConstructor(object):
    def __init__(self):
        self.no_count = 1

    def create_network(self, net, input_data):
        net.create_network(input_data)
        net.make_gclstm_layer(32, if_save=True)
        net.make_gclstm_layer(32)
        net.make_hierarchical_network_pooling54()
        net.make_gclstm_layer(16)
        net.make_hierarchical_network_pooling14()
        net.make_gclstm_layer(8, if_concat=True)  #
        net.make_fc_layer(2, name='final', with_bn=False, with_act_func=False)
        print('no_count:', self.no_count)
        self.no_count = self.no_count + 1


class GCLSTMNetConstructor(object):
    def __init__(self):
        self.no_count = 1

    def create_network(self, net, input_data):
        net.create_network(input_data)
        net.make_gclstm_layer(32)
        net.make_gclstm_layer(32)
        net.make_graph_embed_pooling(no_vertices=54)
        net.make_gclstm_layer(16)
        net.make_graph_embed_pooling(no_vertices=14)
        net.make_gclstm_layer(8, if_concat=True)
        net.make_fc_layer(2, name='final', with_bn=False, with_act_func=False)
        print('no_count:', self.no_count)
        self.no_count = self.no_count + 1


def train_model(proportion, atlas, node_number, window_size, step, constructor, train_iterations):
    input_data_process = fmri_pre_proc.HCMDDPreProcess(proportion=proportion, atlas=atlas, node_number=node_number,
                                                       window_size=window_size, step=step, data_dir='data_dir')
    data_set = input_data_process.compute_graph_cnn_input()

    random_s = np.array([25, 50, 100, 125, 150, 175, 200, 225, 250, 275], dtype=int)
    run_experiment(data_set, constructor, node_number, proportion, train_iterations, 'HC_MDD', random_s)

def run_experiment(data_set, constructor, node_number, proportion, train_iterations, name, random_s):
    acc_set = np.zeros((iter_time, 1))
    std_set = np.zeros((iter_time, 1))
    sen_set = np.zeros((iter_time, 1))
    sen_std_set = np.zeros((iter_time, 1))
    spe_set = np.zeros((iter_time, 1))
    spe_std_set = np.zeros((iter_time, 1))
    attr_set = []
    for iter_num in range(iter_time):
        # Decay value for BatchNorm layers, seems to work better with 0.3
        GraphCNNGlobal.BN_DECAY = 0.3

        exp = GraphCNNExperiment('HC_MDD', 'gcn_lstm', constructor())

        exp.num_iterations = train_iterations
        exp.train_batch_size = train_batch_size
        exp.optimizer = 'adam'
        exp.debug = True

        exp.preprocess_data(data_set)
        acc, std, mean_sensitivity, std_sensitivity, mean_specificity, std_specificity, attr = exp.run_kfold_experiments(
            no_folds=10,random_state=random_s[iter_num])
        print_ext('10-fold: %.2f (+- %.2f)' % (acc, std))
        print_ext('sensitivity is: %.2f (+- %.2f)' % (mean_sensitivity, std_sensitivity))
        print_ext('specificity is: %.2f (+- %.2f)' % (mean_specificity, std_specificity))

        acc_set[iter_num] = acc
        std_set[iter_num] = std
        sen_set[iter_num] = mean_sensitivity
        sen_std_set[iter_num] = std_sensitivity
        spe_set[iter_num] = mean_specificity
        spe_std_set[iter_num] = std_specificity
        attr_set.append(attr)

    attr_set = np.array(attr_set)
    path = 'results/' + name + '.mat'
    scipy.io.savemat(path, {'attr_set': attr_set})
    acc_mean = np.mean(acc_set)
    acc_std = np.std(acc_set)
    sen_mean = np.mean(sen_set)
    sen_std = np.std(sen_set)
    spe_mean = np.mean(spe_set)
    spe_std = np.std(spe_set)
    print_ext('finish!')
    verify_dir_exists('results/')
    with open('results/10-10_fold_fmri_final.txt', 'a+') as file:
        for iter_num in range(iter_time):
            print_ext('acc %d :    %.2f   sen :    %.2f   spe :    %.2f' % (
                iter_num, acc_set[iter_num], sen_set[iter_num], spe_set[iter_num]))
            file.write('%s\tacc %d :   \t%.2f (+- %.2f)\tsen :   \t%.2f (+- %.2f)\tspe :   \t%.2f (+- %.2f)\n' % (
                str(datetime.now()), iter_num, acc_set[iter_num], std_set[iter_num], sen_set[iter_num],
                sen_std_set[iter_num], spe_set[iter_num], spe_std_set[iter_num]))
        print_ext('acc:     %.2f(+-%.2f)   sen:     %.2f(+-%.2f)   spe:     %.2f(+-%.2f)' % (
            acc_mean, acc_std, sen_mean, sen_std, spe_mean, spe_std))
        file.write('%s\t %.2f acc  :   \t%.2f (+- %.2f)  sen  :   \t%.2f (+- %.2f)  spe  :   \t%.2f (+- %.2f)\n' % (
            str(datetime.now()), proportion, acc_mean, acc_std, sen_mean, sen_std, spe_mean, spe_std))


def main_gcn_lstm():
    proportion = 0.20
    atlas = ''  #
    node_number = 90  #
    # constructor = GCLSTMNetConstructor  #
    window_size = 80
    step = 2
    constructor = GCLSTMHiNetConstructor  #
    train_iterations = 1000

    flag = tf.app.flags
    flag.DEFINE_integer('flag_per_sub_adj_num', 76, 'per_sub_adj_num')  # 66
    flag.DEFINE_integer('node_number', node_number, 'per_sub_adj_num')

    # for proportion in np.arange(0.02, 0.32, 0.02):
    train_model(proportion, atlas, node_number, window_size, step, constructor, train_iterations)

train_batch_size = 30
iter_time = 10
main_gcn_lstm()
