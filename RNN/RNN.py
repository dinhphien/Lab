#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 14:42:18 2018

@author: phien
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class RNN():
    def __init__(self, input_size=1, num_steps=3, lstm_size=16, num_layers=1, num_epoch=100):
        self._input_size = input_size
        self._num_steps = num_steps
        self._lstm_size = lstm_size
        self._num_layers = num_layers
        self._keep_prob = 1.0
        self._num_epoch = num_epoch

    def build_model(self, X_train, Y_train, X_test, Y_test, min_data, max_data):
        # build graph
        tf.reset_default_graph()
        lstm_graph = tf.Graph()
        with lstm_graph.as_default():
            inputs = tf.placeholder(tf.float32, [None, self._num_steps, self._input_size], name="inputs")
            targets = tf.placeholder(tf.float32, [None, self._input_size], name="targets")

            cells = [tf.nn.rnn_cell.LSTMCell(num_units=self._lstm_size) for i in range(self._num_layers)]
            cell_compose = tf.nn.rnn_cell.MultiRNNCell(cells)
            val, state = tf.nn.dynamic_rnn(cell_compose, inputs, dtype=tf.float32)
            # val of shape (batch_size, num_steps, lstm_size)
            val = tf.transpose(val, perm=[1, 0, 2])
            # val of shape (num_steps, batch_size, lstm_size)
            last_val = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")
            # last_val of shape (batch_size, lstm_size)

            weights = tf.Variable(tf.truncated_normal([self._lstm_size, self._input_size]), name="weights")
            bias = tf.Variable(tf.constant(0.1, shape=[self._input_size]), name="bias")
            prediction = tf.matmul(last_val, weights) + bias

            loss = tf.reduce_mean(tf.abs(prediction - targets))
            optimizer = tf.train.AdamOptimizer()
            opt = optimizer.minimize(loss)
            
            # run

        with tf.Session(graph=lstm_graph) as sess:
            sess.run(tf.global_variables_initializer())

            loss_trains = []
            loss_tests = []
            for epoch in range(self._num_epoch):
                total_loss_train = 0
                for i in range(X_train.shape[0]):

                    train_data_feed = {
                        inputs: X_train[i],
                        targets: Y_train[i],
                    }
                    loss_tr, _ = sess.run([loss, opt],feed_dict=train_data_feed)
                    total_loss_train += loss_tr
                loss_trains.append(total_loss_train / X_train.shape[0])

                vali_data_feed = {
                    inputs: X_test,
                    targets: Y_test
                }
                loss_vali, = sess.run([loss], vali_data_feed)
                loss_tests.append(loss_vali)
                # print(pre_.shape)
                # y_pred = pre_ * (max_data - min_data) + min_data


                # loss_val = np.mean(np.abs(pre_, Y_test))

                print("Epoch : %d train_loss: %f vali_loss: %f" %(epoch,(total_loss_train/ X_train.shape[0]), loss_vali ))

            iters = np.arange(self._num_epoch)
            plt.plot(iters, loss_trains,"g-")
            plt.plot(iters, loss_tests, "r-")
            plt.show()





