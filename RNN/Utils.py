import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data(data_file_path):
    df = pd.read_csv(data_file_path, header=None)
    print("Loaded data from :" + data_file_path)
    data = df.values[:, 3]
    print(data.shape)
    # plt.plot(df.values[:, 0 ], data)
    # plt.show()
    return data                                      # return numpy array data of shape (N, )


def split_data(data, ratio_train=0.8):                # data type: n-D array, ratio: the ratio of train_data and total_data
    index_split = int(data.shape[0] * ratio_train)
    train_data = data[:index_split]
    test_data = data[index_split:]
    return train_data, test_data               # return train_data, test_data


def normailize_data(data):                     # data n_D numpy array
    min_data = np.min(data)
    max_data = np.max(data)
    data = (data - min_data) / (max_data - min_data)
    return data, min_data, max_data                     # return n_D numpy array and instance of MinMaxScaler: scale


def un_normalize_data(data, max_data, min_data):           # data:n-D numpy array
    data = data * (max_data - min_data) + min_data
    return data

