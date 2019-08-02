from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import logging
import time
import math
import os
class NTNCNN(object):

    def __init__(self,
                 session,
                 logger,
                 data_type,
                 hidden_dim=100,
                 kprob=1.0,
                 batch_size=128,
                 epoch=200,
                 lr=0.005,
                 lambda_value=0.,
                 n_class=2,
                 window_size=3,
                 thres=.5,
                 n_w=1.,
                 p_w=1.
                 ):

        """
        :param session: session
        :param data_type: '3Q', 'Ran', 'PC'
        :param hidden_dim: The number of hidden nodes
        :param kprob: 1-(deop_out)
        :param batch_size: minibatch_size
        :param epoch: epoch
        :param lr: learning rate
        :param lambda_value: regularizer
        :param n_class: the number of class to classify
        :param window_size: cnn kernel size
        :param thres: threshold
        :param n_w: not use (weight on class 1)
        :param p_w: not use (weight on class 0)

        To differentiate Class instances and Methods's instances,
        we use 'self._' and 'self.'
        """
        self._n_class = n_class
        self._hidden_dim = hidden_dim
        self._window = window_size
        self._kprob = kprob
        self._batch_size = batch_size
        self._epoch = epoch
        self._lr = lr
        self._lambda_value = lambda_value
        self._session = session
        self._logger = logger
        self._thres = thres
        self._n_w = n_w
        self._p_w = p_w
        self._data_type = data_type

        self._params = {'data_type': data_type, 'hidden_dim': hidden_dim, 'kprob': kprob, 'lambda_value': lambda_value,
                        'batch_size': batch_size, 'epoch': epoch, 'lr': lr}
        self._graph_exist = False

    def Conv_fn(self, input_dt, neighbor, depth=1, name='Conv'):
        """

        :param input_dt: Input data
        :param neighbor: window size
        :param depth: the number of filter
        :param name: conv name
        :return: Maxpooled data
        """
        input_layer = tf.expand_dims(input_dt, -1)
        conv = tf.layers.conv2d(inputs=input_layer,
                                filters=depth,
                                kernel_size=[neighbor, 1],
                                strides=1,
                                padding='VALID',
                                trainable=True,
                                activation=tf.nn.relu,
                                name=name)
        maxpool_flat = tf.reduce_max(tf.squeeze(conv,3), axis=1)
        return maxpool_flat

    def Add_Hidden_Layer(self, hidden_input, hidden_dim, kprob, activation='relu', dropout=False, num=1):
        """

        :param hidden_input: Input
        :param hidden_dim: the number of nodes on hidden layer
        :param kprob: 1=(dropout)
        :param activation: activation fn.
        :param dropout: boolean
        :param num: identity of each hidden layer
        :return: Hidden layer
        """
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.name_scope('hidden_layer'):
            w = tf.get_variable(shape=[int(hidden_input.shape[1]), hidden_dim], initializer=initializer,
                                dtype=tf.float32, name='hidden_weight{}'.format(num))
            b = tf.get_variable(shape=[hidden_dim,], initializer=initializer,
                                dtype=tf.float32, name='hidden_bias{}'.format(num))
            hidden = tf.matmul(hidden_input, w) + b

            # activation fn.
            if activation == 'None':
                pass
            elif activation == 'relu':
                hidden = tf.nn.relu(hidden, name='hidden{}'.format(num))
            elif activation == 'tanh':
                hidden = tf.tanh(hidden, name='hidden{}'.format(num))

            if dropout == True:
                return tf.nn.dorpout(hidden, kprob, name="hidden_drop{}".format(num))
            else:
                return hidden

    def build_graph(self):

        """
        Draw Graph

        :return: All tensor graph
        """
        keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')
        self.keep_prob = keep_prob

        U_l = tf.placeholder(tf.float32, [None, 22, self._eev_dim], name='U_l')
        U_m = tf.placeholder(tf.float32, [None, 5, self._eev_dim], name='U_m')
        U_s = tf.placeholder(tf.float32, [None, 1, self._eev_dim], name='U_s')
        y = tf.placeholder_with_default([0., 0.], [None, self._n_class], name='y')

        self.U_l = U_l
        self.U_m = U_m
        self.U_s = U_s
        self.y = y

        V_short = tf.squeeze(U_s, axis=1)
        start_num = 1

        V_long = self.Conv_fn(input_dt=U_l, neighbor=self._window, depth=1, name='Conv_Long')
        V_mid = self.Conv_fn(input_dt=U_m, neighbor=self._window, depth=1, name='Conv_Mid')
        V_flat = tf.concat([V_long, V_mid, V_short], axis=1, name='V_flat')

        last_dim = self._hidden_dim//2
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.name_scope('last_layer'):
            w = tf.get_variable(shape=[last_dim, self._n_class], initializer=initializer,
                                dtype=tf.float32, name='last_weight')
            b = tf.get_variable(shape=[self._n_class,], initializer=initializer,
                                dtype=tf.float32, name='last_bias')

        h = self.Add_Hidden_Layer(V_flat, self._hidden_dim, activation='relu', dropout=True, kprob=self.keep_prob,
                                  num=start_num+1)
        h_last = self.Add_Hidden_Layer(V_flat, self._hidden_dim, activation='relu', dropout=True, kprob=self.keep_prob,
                                  num=start_num + 2)

        logits = tf.matmul(h_last, w) + b
        pred_prob = tf.nn.softmax(logits, name='pred_prob')

        weight = tf.matmul(tf.cast(y, tf.float32), tf.constant([[self._n_w], [self._p_w]]))
        weight = tf.squeeze(weight)
        loss = tf.losses.softmax_cross_entropy(y, logits, weights=weight)

        train_vars = tf.trainable_variables()
        regularizer = self._lambda_value * tf.add_n([tf.nn.l2_loss(v) for v in train_vars])
        regularized_loss = tf.add(loss, regularizer)
        self.regularized_loss = regularized_loss

        optimizer = tf.train.AdamOptimizer(self._lr).minimize(regularized_loss)
        accuracy, conf_mat = self.Accuracy(pred_prob, self._thres)

        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(i)

        self.optimizer = optimizer
        self.loss = loss
        self.accuracy = accuracy
        self.conf_mat = conf_mat
        self.pred_prob = pred_prob
        self.V_flat = V_flat
        self.h = h
        self.logits = logits
        self.saver = tf.train.Saver()
        self.h_last = h_last

        self._graph_exist = True

    def Accuracy(self, pred, y, thres):
        """
        get accuracy
        :param pred: predicted probability
        :param y: true y
        :param thres: cutoff value
        :return: Accuracy, confusion matrix
        """

        y_bool = tf.equal(y[:,1],1)
        pred_bool = pred[:,1] > thres
        is_correct = tf.equal(pred_bool, y_bool)
        TN = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_bool),tf.logical_not(pred_bool)), tf.int64))
        TP = tf.reduce_sum(tf.cast(tf.logical_and(y_bool, pred_bool), tf.int64))
        FN = tf.reduce_sum(tf.cast(tf.logical_and(y_bool, tf.logical_not(pred_bool)), tf.int64))
        FP = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_bool), pred_bool), tf.int64))
        conf_mat = {'TN': TN, 'TP':TP, 'FN': FN, 'FP': FP}
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        return accuracy, conf_mat

    def train(self, train_U_l, train_y, val_U_l, val_y):
        """
        train fn.
        :param train_U_l:
        :param train_Y:
        :param val_U_l:
        :param val_Y:
        :return: History of train procedure
        """
        train_U_m = train_U_l[:, :5, :]
        train_U_s = train_U_l[:, :1, :]

        val_U_m = val_U_l[:, :5, :]
        val_U_s = val_U_l[:, :1, :]

        self._eev_dim = train_U_l.shape[-1]

        if not self._graph_exist:
            self.build_graph()


        """ training """
        logger = self._logger
        logger.info('Global Variables are  Initialized')
        self._session.run(tf.global_variables_initializer())

        # feed dictionary ======================================

        train_dict = {self.U_l: train_U_l,
                      self.U_m: train_U_m,
                      self.U_s: train_U_s,
                      self.y: train_y}

        val_dict = {self.U_l: val_U_l,
                      self.U_m: val_U_m,
                      self.U_s: val_U_s,
                      self.y: val_y}
        # ======================================================
        n_train = len(train_U_l)

        train_acc = []
        val_acc = []

        train_loss = []
        train_reg_loss = []
        val_loss = []

        n_es = 0
        start_time = time.time()
        for epoch in range(self._epoch):
            batch_size = self._batch_size
            idx_set = np.arange(n_train)
            iter_num = math.ceil(n_train / batch_size)
            for i in range(iter_num):
                if idx_set.size == 1:
                    break
                elif idx_set.size < self._batch_size:
                    batch_size = idx_set.size
                else:
                    pass
                idx_used = np.random.choice(idx_set, size=batch_size, replace=False)
                batch_dict = {self.U_l: train_U_l[idx_used],
                              self.U_m: train_U_m[idx_used],
                              self.U_s: train_U_s[idx_used],
                              self.y: train_y[idx_used],
                              self.keep_prob: self._kprob}

                _, reg_loss_val, loss_val = self._session.run([self.optimizer, self.regularized_loss, self.loss],
                                                              feed_dict=batch_dict)
                idx_set = np.setdiff1d(idx_set, idx_used)

            # for each epoch
            t_acc, t_reg_loss, t_loss = self._session.run([self.optimizer, self.regularized_loss, self.loss],
                                                              feed_dict=train_dict)
            train_acc.append(t_acc)
            train_loss.append(t_loss)
            train_reg_loss.append(t_reg_loss)

            v_acc, v_loss = self._session.run([self.accuracy, self.loss], feed_dict=val_dict)
            val_acc.append(v_acc)
            val_loss.append(v_loss)

            earlystop = (epoch > 30) and (min(train_reg_loss) == train_reg_loss[-1])
            earlystop = earlystop and (min(train_reg_loss)/sorted(set(train_reg_loss))[1] >= 0.9990)
            if earlystop:
                n_es += 1
            else:
                n_es = 0

            if n_es >= 2:
                print("early_stop !!")
                break
        self._params['last_epoch'] = epoch + 1

        history = {'train_acc': train_acc,
                   'train_loss': train_loss,
                   'train_reg_loss': train_reg_loss,
                   'val_acc': val_acc,
                   'val_loss': val_loss,
                   'last_epoch': self._params['last_epoch']}
        self.history = history

    def predict(self, U_l):

        """
        Calculate y hat
        :param U_l: Input
        :return: y hat (shape = N by 2 matrix. 1st col is 0 prob, and 2nd col is 1 prob
        """
        try:
            U_m = U_l[:, :5, :]
            U_s = U_l[:, :1, :]
        except:
            U_l = np.expand_dims(U_l, axis=0)
            U_m = U_l[:, :5, :]
            U_s = U_l[:, :1, :]
        new_dict = {self.U_l: U_l, self.U_m: U_m, self.U_s: U_s}
        pred_y = self._session.run(self.pred_prob, feed_dict=new_dict)

        return pred_y

    def save(self, save_dir):
        """
        Save the model
        :param save_dir: where to svae
        :return: model.ckpt file
        """

        ckpt_path = os.path.join(save_dir, 'prediction_model.ckpt')
        self.saver.save(self._session, ckpt_path)

    def give(self, U_l):

        """
        Calculate last feature
        :param U_l: Input
        :return: y hat (shape = N by 2 matrix. 1st col is 0 prob, and 2nd col is 1 prob
        """
        try:
            U_m = U_l[:, :5, :]
            U_s = U_l[:, :1, :]
        except:
            U_l = np.expand_dims(U_l, axis=0)
            U_m = U_l[:, :5, :]
            U_s = U_l[:, :1, :]
        new_dict = {self.U_l: U_l, self.U_m: U_m, self.U_s: U_s}
        h_value = self._session.run(self.h_last, feed_dict=new_dict)

        return h_value

def my_logger(logger_name, log_save_path, file_names):
    """
    Make log file
    :param logger_name:
    :param log_save_path:
    :param file_names:
    :return: logger
    """
    logger = logging.getLogger(logger_name)
    handler = logging.FileHandler(log_save_path + '/' + file_names + '.log')
    formatter = logging.formater('%(asctime)s %(levelnames)s %(message)s ')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # for console print
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)
    return logger