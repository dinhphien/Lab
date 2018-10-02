#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 20:32:01 2018

@author: phien
"""

import U
from DataCreator import DataGeneratorSeq
from RNN import RNN

dataset= Utils.load_data("~/AI_AnhThieu/ai/data/GoogleTrace/data_resource_usage_5Minutes_6176858948.csv")
data, min_data, max_data = Utils.normailize_data(dataset)
# print(data[:10])
data_gen = DataGeneratorSeq(data)
X, Y = data_gen.create_batch_data(input_size=1, num_steps=3, batch_size=64)
X_train, X_test = Utils.split_data(X, ratio_train=0.8)
Y_train, Y_test = Utils.split_data(Y, ratio_train=0.8)
# print(X_train.shape[0])
# print(X_train.shape)
# lol
X_val = X_test.reshape(-1, 3 , 1)
Y_val = Y_test.reshape(-1, 1)
rnn = RNN(input_size=1, num_steps=3, lstm_size= 8, num_layers= 1, num_epoch= 10)
rnn.build_model(X_train, Y_train, X_val, Y_val, min_data, max_data)

