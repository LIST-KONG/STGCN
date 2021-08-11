from graphcnn.layers import *
import tensorflow as tf2
import numpy as np
from tensorflow.python.ops.rnn import dynamic_rnn
import graphcnn.setup.dti_pre_process as pre_process
import graphcnn.gc_lstm as gc_lstm

no_ite = 0
FLAGS = tf.app.flags.FLAGS


class GraphCNNNetwork(object):
    def __init__(self):
        self.current_V = None
        self.current_A = None
        self.current_mask = None
        self.labels = None
        self.pooling_weight54 = None
        self.pooling_weight14 = None
        self.network_debug = False
        self.attr = None

    def create_network(self, input):
        self.current_V = input[0]
        self.current_A = input[1]
        self.labels = input[2]
        self.current_mask = None  # input[3]
        self.current_V = tf.reshape(input[0],
                                    [-1, FLAGS.flag_per_sub_adj_num, FLAGS.node_number, input[0].get_shape()[3]])
        self.current_A = tf.reshape(input[1],
                                    [-1, FLAGS.flag_per_sub_adj_num, FLAGS.node_number, input[1].get_shape()[3],
                                     FLAGS.node_number])
        pooling_weight54, pooling_weight14 = pre_process.compute_pooling_weight()
        self.pooling_weight54 = tf.constant(pooling_weight54, dtype=tf.float32)
        self.pooling_weight14 = tf.constant(pooling_weight14, dtype=tf.float32)
        if self.network_debug:
            size = tf.reduce_sum(self.current_mask, axis=1)
            self.current_V = tf.Print(self.current_V,
                                      [tf.shape(self.current_V), tf.reduce_max(size), tf.reduce_mean(size)],
                                      message='Input V Shape, Max size, Avg. Size:')

        return input

    def make_batchnorm_layer(self):
        if self.current_mask != None:
            self.current_mask = tf2.Print(self.current_mask, [tf2.shape(self.current_mask)],
                                          message="current_mask is the size:", summarize=4)
        self.current_V = make_bn(self.current_V, self.is_training, mask=self.current_mask, num_updates=self.global_step)
        return self.current_V

    # Equivalent to 0-hop filter
    def make_embedding_layer(self, no_filters, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='Embed') as scope:
            self.current_V = make_embedding_layer(self.current_V, no_filters)
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
        return self.current_V, self.current_A, self.current_mask

    def make_dropout_layer(self, keep_prob=0.5):
        self.current_V = tf.cond(self.is_training, lambda: tf.nn.dropout(self.current_V, keep_prob=keep_prob),
                                 lambda: self.current_V)
        return self.current_V

    def make_graphcnn_layer(self, no_filters, no_count, i, name=None, with_bn=True, with_act_func=True):
        global no_ite
        with tf.variable_scope(name, default_name='Graph-CNN') as scope:
            self.current_V = make_graphcnn_layer(self.current_V, self.current_A, no_filters)
            # self.current_V = tf2.Print(self.current_V, [tf2.shape(self.current_V)], message="curren_V is the size:",summarize=4)
            if self.current_mask != None:
                self.current_mask = tf2.Print(self.current_mask, [tf2.shape(self.current_mask)],
                                              message="current_mask is the size:", summarize=4)
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
            if self.network_debug:
                batch_mean, batch_var = tf.nn.moments(self.current_V, np.arange(len(self.current_V.get_shape()) - 1))
                self.current_V = tf.Print(self.current_V, [tf.shape(self.current_V), batch_mean, batch_var],
                                          message='"%s" V Shape, Mean, Var:' % scope.name)
        return self.current_V

    def make_graph_embed_pooling(self, no_vertices=1, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='GraphEmbedPool') as scope:
            V_shape = self.current_V.get_shape()
            A_shape = self.current_A.get_shape()
            reshape_V = tf.reshape(self.current_V, (-1, V_shape[2], V_shape[3]))
            reshape_A = tf.reshape(self.current_A, (-1, A_shape[2], A_shape[3], A_shape[4]))
            self.current_V, self.current_A = make_graph_embed_pooling(reshape_V, reshape_A,
                                                                      mask=self.current_mask, no_vertices=no_vertices)
            self.current_V = tf.reshape(self.current_V, (-1, V_shape[1], no_vertices, V_shape[3]))
            self.current_A = tf.reshape(self.current_A, (-1, A_shape[1], no_vertices, A_shape[3], no_vertices))

            self.current_mask = None
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
            if self.network_debug:
                batch_mean, batch_var = tf.nn.moments(self.current_V, np.arange(len(self.current_V.get_shape()) - 1))
                self.current_V = tf.Print(self.current_V, [tf.shape(self.current_V), batch_mean, batch_var],
                                          message='Pool "%s" V Shape, Mean, Var:' % scope.name)
        return self.current_V, self.current_A, self.current_mask

    def make_hierarchical_network_pooling54(self, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='HierarchicalPool') as scope:
            factors = self.pooling_weight54

            V_shape = self.current_V.get_shape()
            reshape_V = tf.reshape(self.current_V, (-1, V_shape[2], V_shape[3]))
            reshape_V = batch_matmul(reshape_V, factors)
            self.current_V = tf.reshape(reshape_V, (-1, V_shape[1], 54, V_shape[3]))
            # self.current_V = tf.Print(self.current_V, [tf.shape(self.current_V)])
            # self.current_A = tf.Print(self.current_A, [tf.shape(self.current_A)])

            A_shape = self.current_A.get_shape()
            result_A = tf.reshape(self.current_A, (-1, A_shape[-1]))
            result_A = tf.matmul(result_A, factors)
            result_A = tf.reshape(result_A, (-1, A_shape[2], A_shape[3] * 54))
            result_A = batch_matmul(result_A, factors)
            result_A = tf.reshape(result_A, (-1, A_shape[1], 54, A_shape[3], 54))
            self.current_A = result_A

            self.current_mask = None
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
            if self.network_debug:
                batch_mean, batch_var = tf.nn.moments(self.current_V, np.arange(len(self.current_V.get_shape()) - 1))
                self.current_V = tf.Print(self.current_V, [tf.shape(self.current_V), batch_mean, batch_var],
                                          message='Pool "%s" V Shape, Mean, Var:' % scope.name)
        return self.current_V, self.current_A, self.current_mask

    def make_hierarchical_network_pooling14(self, name=None, with_bn=True, with_act_func=True):
        with tf.variable_scope(name, default_name='HierarchicalPool') as scope:
            factors = self.pooling_weight14

            V_shape = self.current_V.get_shape()
            reshape_V = tf.reshape(self.current_V, (-1, V_shape[2], V_shape[3]))
            reshape_V = batch_matmul(reshape_V, factors)
            self.current_V = tf.reshape(reshape_V, (-1, V_shape[1], 14, V_shape[3]))

            A_shape = self.current_A.get_shape()
            result_A = tf.reshape(self.current_A, (-1, A_shape[-1]))
            result_A = tf.matmul(result_A, factors)
            result_A = tf.reshape(result_A, (-1, A_shape[2], A_shape[3] * 14))
            result_A = batch_matmul(result_A, factors)
            result_A = tf.reshape(result_A, (-1, A_shape[1], 14, A_shape[3], 14))
            self.current_A = result_A

            self.current_mask = None
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
            if self.network_debug:
                batch_mean, batch_var = tf.nn.moments(self.current_V, np.arange(len(self.current_V.get_shape()) - 1))
                self.current_V = tf.Print(self.current_V, [tf.shape(self.current_V), batch_mean, batch_var],
                                          message='Pool "%s" V Shape, Mean, Var:' % scope.name)
        return self.current_V, self.current_A, self.current_mask

    def make_fc_layer(self, no_filters, name=None, with_bn=False, with_act_func=True):
        with tf.variable_scope(name, default_name='FC') as scope:
            self.current_mask = None

            if len(self.current_V.get_shape()) >= 2:
                no_input_features = int(np.prod(self.current_V.get_shape()[1:]))  # change
                self.current_V = tf.reshape(self.current_V, [-1, no_input_features])
            self.current_V = make_embedding_layer(self.current_V, no_filters)
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
        return self.current_V

    def make_cnn_layer(self, no_filters, name=None, with_bn=False, with_act_func=True, filter_size=3, stride=1,
                       padding='SAME'):
        with tf.variable_scope(None, default_name='conv') as scope:
            dim = self.current_V.get_shape()[-1]
            kernel = make_variable_with_weight_decay('weights',
                                                     shape=[filter_size, filter_size, dim, no_filters],
                                                     stddev=math.sqrt(1.0 / (no_filters * filter_size * filter_size)),
                                                     wd=0.0005)
            conv = tf.nn.conv2d(self.current_V, kernel, [1, stride, stride, 1], padding=padding)
            biases = make_bias_variable('biases', [no_filters])
            self.current_V = tf.nn.bias_add(conv, biases)
            if with_bn:
                self.make_batchnorm_layer()
            if with_act_func:
                self.current_V = tf.nn.relu(self.current_V)
            return self.current_V

    def make_pool_layer(self, padding='SAME'):
        with tf.variable_scope(None, default_name='pool') as scope:
            dim = self.current_V.get_shape()[-1]
            self.current_V = tf.nn.max_pool(self.current_V, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding=padding,
                                            name=scope.name)
            return self.current_V

    def make_gclstm_layer(self, no_filters, if_concat=False, name=None, if_save=False):
        with tf.variable_scope(name, default_name='gclstm') as scope:
            self.current_mask = None
            # self.current_V = tf.Print(self.current_V, [self.current_V[0, 0, 0, :]], message="V1: ")
            # self.current_V = tf.Print(self.current_V, [self.current_V[1, 0, 0, :]], message="V2: ")

            self.current_V, attr = gc_lstm.gcnlstm_loop(lstm_size=FLAGS.flag_per_sub_adj_num,
                                                        input_data_V=self.current_V,
                                                        input_data_A=self.current_A, no_filter=no_filters,
                                                        if_concat=if_concat)
            if if_save:
                self.attr = attr

        return self.current_V
