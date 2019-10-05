import torch
from torch.autograd import Variable
import torch.nn as nn


'''
使用深度神经网络降维
'''

class AutoEncoders():
    def __init__(self, learing_rate, hidden_layers, epoch, batch_size, save_path):
        pass

    def fit(self, X_train, y_train, X_test, y_test):
        ''' 这里的 y 是不需要用来训练的，只是为了后续的索引合并 '''
        pass

    def save(self):
        pass


class VAE():
    def __init__(self):
        pass

    def fit(self, X_train, y_train, X_test, y_test):
        ''' 这里的 y 是不需要用来训练的，只是为了后续的索引合并 '''
        pass

    def save(self):
        pass
