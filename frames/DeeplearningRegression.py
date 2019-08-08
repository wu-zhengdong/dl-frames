import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from . import tools
from .Based_model import conv_bn_net
from .Based_model import lstm_network


class ANN():
    def __init__(self, hidden_layers, learning_rate, dropout=0, activate_function='relu', epoch=2000, batch_size=128):
        # self.layers = layers
        self.hidden_layers = hidden_layers
        self.lr = learning_rate
        self.dropout = dropout
        self.activate_function = activate_function
        self.epoch = epoch
        self.batch_size = batch_size

        self.TrainLosses = []

    def model(self, input_size, output_size):
        ''' 搭建网络 '''
        hidden_layers = self.hidden_layers
        layers = []

        # 判断 激活函数
        if self.activate_function == 'relu':
            activate_function = nn.ReLU()
        if self.activate_function == 'sigmoid':
            activate_function = nn.Sigmoid()
        if self.activate_function == 'tanh':
            activate_function = nn.Tanh()

        # input_layer
        input_layer = nn.Linear(input_size, hidden_layers[0])
        layers.append(input_layer)
        layers.append(activate_function)
        layers.append(nn.Dropout(self.dropout))

        # hidden layers
        hidden_layers_number = len(hidden_layers)  # 隐藏层个数

        for i in range(hidden_layers_number):
            try:
                layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
                layers.append(activate_function)
                layers.append(nn.Dropout(self.dropout))
            except:
                pass

        # output layer
        output_layer = nn.Linear(hidden_layers[-1], output_size)
        layers.append(output_layer)

        # create network 找不到好办法，只能用 if else 去判断网络层数来搭建，这样的缺陷是：网络不能动态调整
        if hidden_layers_number == 1:
            seq_net = nn.Sequential(
                layers[0], layers[1], layers[2], layers[4]
            )

        if hidden_layers_number == 2:
            seq_net = nn.Sequential(
                layers[0], layers[1], layers[2], layers[3], layers[4], layers[5], layers[6]
            )
        if hidden_layers_number == 3:
            seq_net = nn.Sequential(
                layers[0], layers[1], layers[2], layers[3], layers[4], layers[5], layers[6], layers[7], layers[8], layers[9]
            )

        if hidden_layers_number == 4:
            seq_net = nn.Sequential(
                layers[0], layers[1], layers[2], layers[3], layers[4], layers[5], layers[6], layers[7], layers[8],
                layers[9], layers[10], layers[11], layers[12], layers[13]
            )

        if hidden_layers_number == 5:
            seq_net = nn.Sequential(
                layers[0], layers[1], layers[2], layers[3], layers[4], layers[5], layers[6], layers[7], layers[8],
                layers[9], layers[10], layers[11], layers[12], layers[13], layers[14], layers[15], layers[16], layers[17], layers[18]
            )

        return seq_net

    def create_batch_size(self, X_train, y_train):

        datasets = torch.FloatTensor(np.c_[X_train, y_train])
        batch_train_set = DataLoader(datasets, batch_size=self.batch_size, shuffle=True)
        return batch_train_set

    def fit(self, X_train, y_train):
        '''
        :param X_train:
        :param y_train:
        :return:
        '''
        # if y is a scalar
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        input_size, output_size = X_train.shape[1], y_train.shape[1]
        batch_train_set = self.create_batch_size(X_train, y_train)

        self.net = self.model(input_size, output_size)
        self.net.train()
        try:
            optim = torch.optim.Adam(self.net.parameters(), lr=self.lr[0], weight_decay=1e-8)
        except:
            optim = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-8)
        criterion = torch.nn.MSELoss()

        for e in range(self.epoch):

            # 是否使用 梯度衰减
            try:
                if (self.epoch + 1) % (self.epoch // 3) == 0:
                    optim.param_groups[0]["lr"] = self.lr[1]
                if (self.epoch + 1) % (self.epoch // 3 * 2) == 0:
                    optim.param_groups[0]["lr"] = self.lr[2]
            except:
                pass

            for b_train in batch_train_set:
                if torch.cuda.is_available():
                    #print('cuda')
                    self.net = self.net.cuda()
                    train_x = Variable(b_train[:, :-1]).cuda()
                    train_y = Variable(b_train[:, -1:]).cuda()
                else:
                    train_x = Variable(b_train[:, :-1])
                    train_y = Variable(b_train[:, -1:])

                prediction = self.net(train_x)

                loss = criterion(prediction, train_y)

                optim.zero_grad()
                loss.backward()
                optim.step()

            self.TrainLosses.append(loss.cpu().data.numpy())

            if (e + 1) % 100 == 0:
                print('Training... epoch: {}, loss: {}'.format((e+1), loss.cpu().data.numpy()))

        print('Training completed!')

    def predict(self, X_test):

        self.net.eval()
        if torch.cuda.is_available():
            test_x = Variable(torch.FloatTensor(X_test)).cuda()
            prediction = self.net(test_x).cpu()
        else:
            test_x = Variable(torch.FloatTensor(X_test))
            prediction = self.net(test_x)

        return prediction

    def score(self, X_test, y_test):

        prediction = self.predict(X_test).data.numpy()
        self.mse, self.rmse, self.mae, self.r2 = tools.calculate(y_test, prediction)

    def result_plot(self, X_test, y_test, save_file, is_show=False):

        prediction = self.predict(X_test).data.numpy()
        plt.plot(range(len(prediction)), prediction, 'r--', label='prediction')
        plt.plot(range(len(y_test)), y_test, 'b--', label="true")
        plt.legend()
        plt.savefig(save_file)
        if is_show:
            plt.show()
        plt.close()
        print('Save the picture successfully!')

    def loss_plot(self):

        plt.plot(range(len(self.TrainLosses)), self.TrainLosses, 'r', label="loss")
        plt.legend()
        plt.show()

    def save_result(self, save_path, is_standard=False, is_PCA=False):
        layer_numbers = len(self.hidden_layers)
        hidden_layers = str(self.hidden_layers).replace(',', '')
        try:
            lr = str(self.lr).replace(',', '')
        except:
            lr = self.lr
        tools.save_ann_results(self.epoch, self.batch_size, lr, self.dropout, layer_numbers, hidden_layers,
                           self.activate_function, self.mse, self.rmse, self.mae, self.r2, is_standard, is_PCA, save_path)
        print('Save results success!')





'''
CNN model
'''


class CNN(object):
    def __init__(self, learning_rate, conv_stride = 1, kernel_size=3, pooling_size=2, pool_stride = 2, channel_numbers = [], flatten = 1024,
                activate_function='relu', dropout=0, epoch=2000, batch_size=128):

        self.conv_stride = conv_stride
        self.kernel_size = kernel_size
        self.pooling_size = pooling_size
        self.pool_stride = pool_stride
        self.channel_numbers = channel_numbers # 每层 conv 的卷积核个数, 即 output_channel
        self.flatten = flatten # flatten 神经元个数

        self.lr = learning_rate
        self.activate_function = activate_function
        self.dropout = dropout
        self.epoch = epoch
        self.batch_size = batch_size

        self.TrainLosses = []

    def conv_padding_same(self, W_in):
        ''' 相当于 tensorflow 里面的 paddling = same 操作 '''
        p = ((W_in - 1) * self.conv_stride + self.kernel_size - W_in) // 2
        return p

    def conv_shape_out(self, input_size):
        '''
        计算卷积池化后的输出矩阵维度，避免小于1，即计算 池化后 的维度，公式：
         (input_size - pooling_sie) // pooling_stride + 1
        '''
        numbers = len(self.channel_numbers)
        out_shape = (input_size - self.pooling_size) // self.pool_stride + 1
        for n in range(numbers-1):
            out_shape = (out_shape - self.pooling_size) // self.pool_stride + 1
        return out_shape

    def conv_model(self, input_size, input_channle):
        ''' 搭建卷积池化层网络 '''
        layers = []

        # 判断 激活函数
        if self.activate_function == 'relu':
            activate_function = nn.ReLU(True)
        if self.activate_function == 'sigmoid':
            activate_function = nn.Sigmoid(True)
        if self.activate_function == 'tanh':
            activate_function = nn.Tanh(True)

        # input layer
        padding = self.conv_padding_same(input_size) # 计算要补几层 0
        conv1 = nn.Conv2d(input_channle, self.channel_numbers[0], kernel_size=self.kernel_size, stride=self.conv_stride, padding=padding)
        pool1 = nn.MaxPool2d(self.pooling_size, self.pool_stride)
        layers.append(conv1)
        layers.append(nn.BatchNorm2d(self.channel_numbers[0]))
        layers.append(activate_function)
        layers.append(nn.Dropout(self.dropout))
        layers.append(pool1)
        W_in = (input_size - 1) // 2 + 1

        # hidden layers
        hidden_layers_number = len(self.channel_numbers)
        for i in range(hidden_layers_number):
            try:
                padding = self.conv_padding_same(W_in)
                layers.append(nn.Conv2d(self.channel_numbers[i], self.channel_numbers[i + 1], kernel_size=self.kernel_size, stride=self.conv_stride, padding=padding))
                layers.append(nn.BatchNorm2d(self.channel_numbers[i + 1]))
                layers.append(activate_function)
                layers.append(nn.Dropout(self.dropout))
                layers.append(nn.MaxPool2d(self.pooling_size, self.pool_stride))
                W_in = (W_in - 1) // 2 + 1
            except:
                pass

        if hidden_layers_number == 1:
            cnn = nn.Sequential(
                layers[0], layers[1], layers[2], layers[3], layers[4]
            )

        if hidden_layers_number == 2:
            cnn = nn.Sequential(
                layers[0], layers[1], layers[2], layers[3], layers[4],
                layers[5], layers[6], layers[7], layers[8], layers[9]
            )

        if hidden_layers_number == 3:
            cnn = nn.Sequential(
                layers[0], layers[1], layers[2], layers[3], layers[4],
                layers[5], layers[6], layers[7], layers[8], layers[9],
                layers[10], layers[11], layers[12], layers[13], layers[14],
            )

        if hidden_layers_number == 4:
            cnn = nn.Sequential(
                layers[0], layers[1], layers[2], layers[3], layers[4],
                layers[5], layers[6], layers[7], layers[8], layers[9],
                layers[10], layers[11], layers[12], layers[13], layers[14],
                layers[15], layers[16], layers[17], layers[18], layers[19],
            )
        return cnn

    def Linear_model(self, out_shape, output_size):
        ''' 搭建全连接层的网路 '''
        # 判断 激活函数
        if self.activate_function == 'relu':
            activate_function = nn.ReLU(True)
        if self.activate_function == 'sigmoid':
            activate_function = nn.Sigmoid(True)
        if self.activate_function == 'tanh':
            activate_function = nn.Tanh(True)

        return nn.Sequential(
            nn.Linear(out_shape, self.flatten),
            activate_function,
            nn.Dropout(self.dropout),
            nn.Linear(self.flatten, output_size)
        )


    def create_batch_size(self, X_train, y_train):

        datasets = torch.FloatTensor(np.c_[X_train, y_train])
        batch_train_set = DataLoader(datasets, batch_size=self.batch_size, shuffle=True)
        return batch_train_set

    def fit(self, X_train, y_train):
        ''' 训练模型 '''
        # if y is a scalar
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        # x_train = [-1, channle, W, H]
        self.W, self.H = int(np.sqrt((X_train.shape[1]))), int(np.sqrt((X_train.shape[1])))

        input_channles, input_size, output_size = 1, self.W, y_train.shape[1]
        self.conv_out = self.conv_shape_out(input_size)

        assert self.conv_out >= 1, '卷积层数过多，超出维度限制'

        batch_train_set = self.create_batch_size(X_train, y_train)

        # 搭建卷积模型
        conv = self.conv_model(self.W, input_channles)
        flatten = self.Linear_model(self.channel_numbers[-1] * self.conv_out**2, output_size)
        self.net = conv_bn_net(conv, flatten)

        self.net.train()
        try:
            optim = torch.optim.Adam(self.net.parameters(), lr=self.lr[0], weight_decay=1e-8)
        except:
            optim = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-8)
        criterion = torch.nn.MSELoss()

        for e in range(self.epoch):

            # 是否使用 梯度衰减
            try:
                if (self.epoch + 1) % (self.epoch // 3) == 0:
                    optim.param_groups[0]["lr"] = self.lr[1]
                if (self.epoch + 1) % (self.epoch // 3 * 2) == 0:
                    optim.param_groups[0]["lr"] = self.lr[2]
            except:
                pass

            for b_train in batch_train_set:
                if torch.cuda.is_available():
                    #print('cuda')
                    self.net = self.net.cuda()
                    train_x = Variable(b_train[:, :-1].view(-1, 1, self.W, self.H)).cuda()
                    train_y = Variable(b_train[:, -1:]).cuda()
                else:
                    train_x = Variable(b_train[:, :-1].view(-1, 1, self.W, self.H))
                    train_y = Variable(b_train[:, -1:])

                prediction = self.net(train_x)

                loss = criterion(prediction, train_y)

                optim.zero_grad()
                loss.backward()
                optim.step()

            self.TrainLosses.append(loss.cpu().data.numpy())

            if (e + 1) % 100 == 0:
                print('Training... epoch: {}, loss: {}'.format((e+1), loss.cpu().data.numpy()))
        print('Training completed!')

    def predict(self, X_test):

        self.net.eval()
        if torch.cuda.is_available():
            test_x = Variable(torch.FloatTensor(X_test).view(-1, 1, self.W, self.H)).cuda()
            prediction = self.net(test_x).cpu()
        else:
            test_x = Variable(torch.FloatTensor(X_test).view(-1, 1, self.W, self.H))
            prediction = self.net(test_x)

        return prediction

    def score(self, X_test, y_test):

        prediction = self.predict(X_test).data.numpy()
        self.mse, self.rmse, self.mae, self.r2 = tools.calculate(y_test, prediction)

    def result_plot(self, X_test, y_test, save_file, is_show=False):

        prediction = self.predict(X_test).data.numpy()
        plt.plot(range(len(prediction)), prediction, 'r--', label='prediction')
        plt.plot(range(len(y_test)), y_test, 'b--', label="true")
        plt.legend()
        plt.savefig(save_file)
        if is_show:
            plt.show()
        plt.close()
        print('Save the picture successfully!')

    def loss_plot(self):

        plt.plot(range(len(self.TrainLosses)), self.TrainLosses, 'r', label="loss")
        plt.legend()
        plt.show()

    def save_result(self, save_path, is_standard=False, is_PCA=False):
        layer_numbers = len(self.channel_numbers)
        hidden_layers = str(self.channel_numbers).replace(',', '')
        try:
            lr = str(self.lr).replace(',', '')
        except:
            lr = self.lr
        tools.save_cnn_results(self.epoch, self.batch_size, lr, self.dropout, layer_numbers, hidden_layers, self.kernel_size
                           , self.conv_stride, self.pooling_size, self.pool_stride, self.flatten, self.activate_function, self.mse,
                           self.rmse, self.mae, self.r2, is_standard, is_PCA, save_path)
        print('Save results success!')



'''
LSTM model
'''


class LSTM():
    def __init__(self, learning_rate, num_layers=2, hidden_size=32, dropout=0, activate_function='relu', epoch=2000, batch_size=128):
        # self.layers = layers
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.dropout = dropout
        self.activate_function = activate_function
        self.epoch = epoch
        self.batch_size = batch_size

        self.TrainLosses = []

    def create_batch_size(self, X_train, y_train):
        p = np.random.permutation(X_train.shape[0])
        data = X_train[p]
        label = y_train[p]

        batch_size = self.batch_size
        batch_len = X_train.shape[0] // batch_size + 1

        b_datas = []
        b_labels = []
        for i in range(batch_len):
            try:
                batch_data = data[batch_size * i: batch_size * (i + 1)]
                batch_label = label[batch_size * i: batch_size * (i + 1)]
            except:
                batch_data = data[batch_size * i: -1]
                batch_label = label[batch_size * i: -1]
            b_datas.append(batch_data)
            b_labels.append(batch_label)

        return b_datas, b_labels

    def fit(self, X_train, y_train):
        '''
        :param X_train:
        :param y_train:
        :return:
        '''
        # if y is a scalar
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        input_size, output_size = X_train.shape[-1], y_train.shape[-1]

        b_data, b_labels = self.create_batch_size(X_train, y_train)

        # 搭建 lstm 网络
        # 判断 激活函数
        if self.activate_function == 'relu':
            activate_function = nn.ReLU(True)
        if self.activate_function == 'sigmoid':
            activate_function = nn.Sigmoid(True)
        if self.activate_function == 'tanh':
            activate_function = nn.Tanh(True)
        self.net = lstm_network(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                output_size=output_size, dropout=self.dropout, activate_function=activate_function)

        self.net.train()
        try:
            optim = torch.optim.Adam(self.net.parameters(), lr=self.lr[0], weight_decay=1e-8)
        except:
            optim = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-8)
        criterion = torch.nn.MSELoss()

        for e in range(self.epoch):

            # 是否使用 梯度衰减
            try:
                if (self.epoch + 1) % (self.epoch // 3 * 2) == 0:
                    optim.param_groups[0]["lr"] = self.lr[1]
                if (self.epoch + 1) % (self.epoch // 3 * 2) == 0:
                    optim.param_groups[0]["lr"] = self.lr[2]
            except:
                pass

            for i in range(len(b_data)):
                if torch.cuda.is_available():
                    #print('cuda')
                    self.net = self.net.cuda()
                    train_x = Variable(torch.FloatTensor(b_data[i])).cuda()
                    train_y = Variable(torch.FloatTensor(b_labels[i])).cuda()
                else:
                    train_x = Variable(torch.FloatTensor(b_data[i]))
                    train_y = Variable(torch.FloatTensor(b_labels[i]))

                prediction = self.net(train_x)

                loss = criterion(prediction, train_y)

                optim.zero_grad()
                loss.backward()
                optim.step()

            self.TrainLosses.append(loss.cpu().data.numpy())

            if (e + 1) % 100 == 0:
                print('Training... epoch: {}, loss: {}'.format((e+1), loss.cpu().data.numpy()))

        print('Training completed!')

    def predict(self, X_test):

        self.net.eval()
        if torch.cuda.is_available():
            test_x = Variable(torch.FloatTensor(X_test)).cuda()
            prediction = self.net(test_x).cpu()
        else:
            test_x = Variable(torch.FloatTensor(X_test))
            prediction = self.net(test_x)

        return prediction

    def score(self, X_test, y_test):

        prediction = self.predict(X_test).data.numpy()
        self.mse, self.rmse, self.mae, self.r2 = tools.calculate(y_test, prediction)

    def result_plot(self, X_test, y_test, save_file, is_show=False):

        prediction = self.predict(X_test).data.numpy()
        plt.plot(range(len(prediction)), prediction, 'r--', label='prediction')
        plt.plot(range(len(y_test)), y_test, 'b--', label="true")
        plt.legend()
        plt.savefig(save_file)
        if is_show:
            plt.show()
        plt.close()
        print('Save the picture successfully!')

    def loss_plot(self):

        plt.plot(range(len(self.TrainLosses)), self.TrainLosses, 'r', label="loss")
        plt.legend()
        plt.show()

    def save_result(self, save_path, is_standard=False, is_PCA=False):
        try:
            lr = str(self.lr).replace(',', '')
        except:
            lr = self.lr
        tools.save_lstm_results(self.epoch, self.batch_size, lr, self.dropout, self.num_layers, self.hidden_size,
                           self.activate_function, self.mse, self.rmse, self.mae, self.r2, is_standard, is_PCA, save_path)
        print('Save results success!')
