#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 10:51:35 2018

@author: phien
"""
import numpy as np



class DataGeneratorSeq:
    def __init__(self, data):
        self._data = data                                # numpy array  of shape (N, )

    def create_batch_data(self, input_size=1, num_steps=3, batch_size=64):
        self._input_size = input_size
        self._num_steps = num_steps
        self._data_length = len(self._data)
        self._segments = self._data_length // self._input_size
        self._batch_size = batch_size
        # generate sequence:
        self._seq = np.array([self._data[i * self._input_size: (i + 1) * self._input_size] for i in range(self._segments)])
        # generate pair X: Y
        self._X = np.array([self._seq[i: i + self._num_steps] for i in range(self._segments - self._num_steps)])
        self._Y = np.array([self._seq[i + self._num_steps] for i in range(self._segments - self._num_steps)])
        # batch_X, batch_Y of shape (batch_size, num_steps, input_size)
        batch_X = np.array([self._X[i * self._batch_size: (i + 1) * self._batch_size] for i in range(len(self._X) // self._batch_size)])
        batch_Y = np.array([self._Y[i * self._batch_size: (i + 1) * self._batch_size] for i in range(len(self._X) // self._batch_size)])

        return batch_X, batch_Y


