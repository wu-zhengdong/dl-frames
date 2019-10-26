import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from . import tools
from .Based_model import conv_bn_net
from .Based_model import lstm_network

import os
import time

'''
The frames of deep learning technology with regression problems, including ANN, CNN, LSTM network.
'''


class ANN():
    def __init__(self, hidden_layers, learning_rate, dropout=0, activate_function='relu', device=0,
                 use_more_gpu=False, epoch=2000, batch_size=128, is_standard=False, weight_decay=1e-8,
                 Dimensionality_reduction_method='None', save_path='ANN_Result'):

        self.save_path = save_path  # 设置一条保存路径，直接把所有的值都收藏起来
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.hidden_layers = hidden_layers
        self.lr = learning_rate
        self.dropout = dropout
        self.activate_function = activate_function
        self.epoch = epoch
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        # 使用第几张 GPU,默认第 0 张，使用单卡。
        self.device = 'cuda:' + str(device)
        # 是否使用多卡
        self.use_more_gpu = use_more_gpu

        # 存储数据预处理的操作
        self.is_standard = is_standard
        self.Dimensionality_reduction_method = Dimensionality_reduction_method

        self.TrainLosses = []
        self.TestLosses = []
        self.t = 0

    def model(self, input_size, output_size):
        ''' create network '''
        hidden_layers = self.hidden_layers
        layers = []

        # 判断 激活函数
        if self.activate_function == 'relu':
            activate_function = nn.ReLU(True)
        if self.activate_function == 'sigmoid':
            activate_function = nn.Sigmoid()
        if self.activate_function == 'tanh':
            activate_function = nn.Tanh()
        if self.activate_function == 'LeakyReLU':
            activate_function = nn.LeakyReLU(True)

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
                layers[0], layers[1], layers[2],
                layers[3]
            )

        if hidden_layers_number == 2:
            seq_net = nn.Sequential(
                layers[0], layers[1], layers[2],
                layers[3], layers[4], layers[5],
                layers[6]
            )
        if hidden_layers_number == 3:
            seq_net = nn.Sequential(
                layers[0], layers[1], layers[2],
                layers[3], layers[4], layers[5],
                layers[6], layers[7], layers[8],
                layers[9]
            )

        if hidden_layers_number == 4:
            seq_net = nn.Sequential(
                layers[0], layers[1], layers[2],
                layers[3], layers[4], layers[5],
                layers[6], layers[7], layers[8],
                layers[9], layers[10], layers[11],
                layers[12]
            )

        if hidden_layers_number == 5:
            seq_net = nn.Sequential(
                layers[0], layers[1], layers[2],
                layers[3], layers[4], layers[5],
                layers[6], layers[7], layers[8],
                layers[9], layers[10], layers[11],
                layers[12], layers[13], layers[14],
                layers[15],
            )

        if hidden_layers_number == 6:
            seq_net = nn.Sequential(
                layers[0], layers[1], layers[2],
                layers[3], layers[4], layers[5],
                layers[6], layers[7], layers[8],
                layers[9], layers[10], layers[11],
                layers[12], layers[13], layers[14],
                layers[15], layers[16], layers[17],
                layers[18]
            )

        return seq_net

    def create_batch_size(self, X_train, y_train):

        datasets = torch.FloatTensor(np.c_[X_train, y_train])
        batch_train_set = DataLoader(datasets, batch_size=self.batch_size, shuffle=True)
        return batch_train_set

    def fit(self, X_train, y_train, X_test, y_test):
        ''' training the network '''
        # if y is a scalar
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)


        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        input_size, output_size = X_train.shape[1], y_train.shape[1]
        batch_train_set = self.create_batch_size(X_train, y_train)

        save_result = os.path.join(self.save_path, 'Results.csv')
        try:
            count = len(open(save_result, 'rU').readlines())
        except:
            count = 1

        net_weight = os.path.join(self.save_path, 'Weight')
        if not os.path.exists(net_weight):
            os.makedirs(net_weight)

        net_path = os.path.join(net_weight, str(count) + '.pkl')
        net_para_path = os.path.join(net_weight, str(count) + '_parameters.pkl')

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Let's use GPU: {}".format(self.device))
        else:
            print("Let's use CPU")

        # 实例化网络
        self.net = self.model(input_size, output_size)
        if self.use_more_gpu and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs")
            # dim = 0 [64, xxx] -> [32, ...], [32, ...] on 2GPUs
            self.net = nn.DataParallel(self.net)
        self.net.to(device)

        try:
            self.net.load_state_dict(torch.load(net_para_path))
        except:
            pass

        self.net.train()
        try:
            optim = torch.optim.Adam(self.net.parameters(), lr=self.lr[0], weight_decay=self.weight_decay)
        except:
            optim = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = torch.nn.MSELoss()

        start = time.time() # 计算时间
        for e in range(self.epoch):

            # 是否使用 梯度衰减
            try:
                if (e + 1) % (self.epoch // 3) == 0:
                    optim.param_groups[0]["lr"] = self.lr[1]
                if (e + 1) % (self.epoch // 3 * 2) == 0:
                    optim.param_groups[0]["lr"] = self.lr[2]
            except:
                pass

            for b_train in batch_train_set:
                if torch.cuda.is_available():
                    #print('cuda')
                    #self.net = self.net.cuda()
                    train_x = Variable(b_train[:, :-1]).to(device)
                    train_y = Variable(b_train[:, -1:]).to(device)
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
                torch.save(self.net, net_path)
                torch.save(self.net.state_dict(), net_para_path)

                self.net.eval()
                if torch.cuda.is_available():
                    test_x = Variable(torch.FloatTensor(X_test)).to(device)
                    test_y = Variable(torch.FloatTensor(y_test)).to(device)
                else:
                    test_x = Variable(torch.FloatTensor(X_test))
                    test_y = Variable(torch.FloatTensor(y_test))

                test_prediction = self.net(test_x)
                test_loss = criterion(test_prediction, test_y)
                self.TestLosses.append(test_loss.cpu().data.numpy())

                self.test_prediction = test_prediction.cpu().data.numpy()
                self.test_prediction[self.test_prediction < 0] = 0

                print('\033[1;35m Testing... epoch: {}, loss: {} \033[0m!'.format((e + 1), test_loss.cpu().data.numpy()))

        end = time.time()
        self.t = end - start
        print('Training completed!!! Time consuming: {}'.format(str(self.t)))

    def predict(self, X_test):

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        self.net.eval()
        if torch.cuda.is_available():
            test_x = Variable(torch.FloatTensor(X_test)).to(device)
        else:
            test_x = Variable(torch.FloatTensor(X_test))

        prediction = self.net(test_x)

        prediction = prediction.cpu().data.numpy()
        prediction[prediction < 0] = 0

        return prediction

    def score(self):

        assert self.test_prediction is not None, '使用 score 前需要 fit'

        self.mse, self.rmse, self.mae, self.mape, \
        self.r2, self.r2_adjusted, self.rmsle = tools.reg_calculate(self.y_test, self.test_prediction,
                                                                    self.X_test.shape[-1])

    def __result_plot(self, save_file):
        
        plt.plot(range(len(self.test_prediction)), self.test_prediction, 'r--', label='prediction')
        plt.plot(range(len(self.y_test)), self.y_test, 'b--', label="true")
        plt.legend()
        try:
            plt.savefig(save_file)
            print('Save the picture successfully!')
        except:
            print('You have not define the path of saving!')
        plt.close()

    def __loss_plot(self, train_save, test_save):

        plt.plot(range(len(self.TrainLosses)), self.TrainLosses, 'r', label="train_loss")
        plt.legend()
        try:
            plt.savefig(train_save)
            print('Save the picture of training loss successfully!')
        except:
            print('You have not define the path of saving!')
        plt.close()

        plt.plot(range(len(self.TestLosses)), self.TestLosses, 'r', label="test_loss")
        plt.legend()
        try:
            plt.savefig(test_save)
            print('Save the picture of testing loss  successfully!')
        except:
            print('You have not define the path of saving!')

        plt.close()

    def __save_result(self, save_path):

        assert self.mse is not None, '需要先调用 score 函数'

        layer_numbers = len(self.hidden_layers)
        hidden_layers = str(self.hidden_layers).replace(',', '')
        try:
            lr = str(self.lr).replace(',', '')
        except:
            lr = self.lr
        count = tools.save_ann_results(self.epoch, self.batch_size, lr, self.dropout,
                                       layer_numbers, hidden_layers, self.activate_function, self.weight_decay,
                                       self.mse, self.rmse,
                                       self.mae, self.mape, self.r2, self.r2_adjusted, self.rmsle, self.is_standard,
                                       self.Dimensionality_reduction_method, t=self.t,
                                       save_result=save_path, train_type='regression')
        print('Save results success!')
        return count

    def save(self):
        '''
        保存 pic， prediction，
        :return:
        '''

        # 保存 result，需要获得一个与行对应的数（count）
        result_path = os.path.join(self.save_path, 'Results.csv')
        count = self.__save_result(result_path)

        # 保存 prediction
        pred_path = os.path.join(self.save_path, 'Prediction')
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
        save_prediction = os.path.join(pred_path, str(count) + '.csv')
        np.savetxt(save_prediction, self.test_prediction, delimiter=',')
        print('Save the value of prediction successfully!!')

        # 保存图片
        pic_path = os.path.join(self.save_path, 'Pictures')
        if not os.path.exists(pic_path):
            os.makedirs(pic_path)
        pic_file = os.path.join(pic_path, str(count) + '.png')
        self.__result_plot(save_file=pic_file)

        # 保存 loss
        train_loss_path_pic = self.save_path + '/Loss/Train/pic'
        test_loss_path_pic = self.save_path + '/Loss/Test/pic'
        train_loss_path_value = self.save_path + '/Loss/Train/Values'
        test_loss_path_value = self.save_path + '/Loss/Test/Values'


        if not os.path.exists(train_loss_path_pic):
            os.makedirs(train_loss_path_pic)

        if not os.path.exists(test_loss_path_pic):
            os.makedirs(test_loss_path_pic)

        if not os.path.exists(train_loss_path_value):
            os.makedirs(train_loss_path_value)

        if not os.path.exists(test_loss_path_value):
            os.makedirs(test_loss_path_value)

        train_loss = os.path.join(train_loss_path_pic, 'train' + str(count) + '.png')
        test_loss = os.path.join(test_loss_path_pic, 'test' + str(count) + '.png')
        self.__loss_plot(train_save=train_loss, test_save=test_loss)

        train_loss_value = np.array(self.TrainLosses)
        test_loss_value = np.array(self.TestLosses)

        np.savetxt(train_loss_path_value + '/train' + str(count) + '.csv', train_loss_value, delimiter=',')
        np.savetxt(test_loss_path_value + '/test' + str(count) + '.csv', test_loss_value, delimiter=',')


'''
CNN model
'''


class CNN(object):
    def __init__(self, learning_rate, conv_stride = 1, kernel_size=3, pooling_size=2, pool_stride = 2,
                 channel_numbers = [], flatten = 1024, activate_function='relu', weight_decay=1e-8, dropout=0, device=0,
                 use_more_gpu=False, epoch=2000, batch_size=128, is_standard=False,
                 Dimensionality_reduction_method='None', save_path='CNN_Results'):

        self.save_path = save_path  # 设置一条保存路径，直接把所有的值都收藏起来
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.conv_stride = conv_stride
        self.kernel_size = kernel_size
        self.pooling_size = pooling_size
        self.pool_stride = pool_stride
        self.channel_numbers = channel_numbers # 每层 conv 的卷积核个数, 即 output_channel
        self.flatten = flatten # flatten 神经元个数

        self.lr = learning_rate
        self.activate_function = activate_function
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.epoch = epoch
        self.batch_size = batch_size

        # 使用第几张 GPU,默认第 0 张，使用单卡。
        self.device = 'cuda:' + str(device)
        # 是否使用多卡
        self.use_more_gpu = use_more_gpu

        self.is_standard = is_standard
        self.Dimensionality_reduction_method = Dimensionality_reduction_method

        self.TrainLosses = []
        self.TestLosses = []

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
            activate_function = nn.Sigmoid()
        if self.activate_function == 'tanh':
            activate_function = nn.Tanh()
        if self.activate_function == 'LeakyReLU':
            activate_function = nn.LeakyReLU(True)

        # input layer
        padding = self.conv_padding_same(input_size) # 计算要补几层 0
        conv1 = nn.Conv2d(input_channle, self.channel_numbers[0], kernel_size=self.kernel_size,
                          stride=self.conv_stride, padding=padding)
        pool1 = nn.MaxPool2d(self.pooling_size, self.pool_stride)
        layers.append(conv1)
        layers.append(nn.BatchNorm2d(self.channel_numbers[0]))
        layers.append(activate_function)
        layers.append(nn.Dropout(self.dropout))
        layers.append(pool1)
        W_in = (input_size - 1) // 2 + 1 # To calculate the shape of matrix after pooling layer.

        # hidden layers
        hidden_layers_number = len(self.channel_numbers)
        for i in range(hidden_layers_number):
            try:
                padding = self.conv_padding_same(W_in)
                layers.append(nn.Conv2d(self.channel_numbers[i], self.channel_numbers[i + 1],
                                        kernel_size=self.kernel_size, stride=self.conv_stride, padding=padding))
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
        if hidden_layers_number == 5:
            cnn = nn.Sequential(
                layers[0], layers[1], layers[2], layers[3], layers[4],
                layers[5], layers[6], layers[7], layers[8], layers[9],
                layers[10], layers[11], layers[12], layers[13], layers[14],
                layers[15], layers[16], layers[17], layers[18], layers[19],
                layers[20], layers[21], layers[22], layers[23], layers[24],
            )
        return cnn

    def Linear_model(self, out_shape, output_size):
        ''' 搭建全连接层的网路 '''
        # 判断 激活函数
        if self.activate_function == 'relu':
            activate_function = nn.ReLU(True)
        if self.activate_function == 'sigmoid':
            activate_function = nn.Sigmoid()
        if self.activate_function == 'tanh':
            activate_function = nn.Tanh()
        if self.activate_function == 'LeakyReLU':
            activate_function = nn.LeakyReLU(True)

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

    def fit(self, X_train, y_train, X_test, y_test):
        ''' 训练模型 '''
        # if y is a scalar
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        # x_train = [-1, channle, W, H]
        self.W, self.H = int(np.sqrt((X_train.shape[1]))), int(np.sqrt((X_train.shape[1])))

        input_channles, input_size, output_size = 1, self.W, y_train.shape[1]
        self.conv_out = self.conv_shape_out(input_size)

        assert self.conv_out >= 1, '卷积层数过多，超出维度限制'

        batch_train_set = self.create_batch_size(X_train, y_train)

        # 搭建卷积模型
        conv = self.conv_model(self.W, input_channles)
        flatten = self.Linear_model(self.channel_numbers[-1] * self.conv_out**2, output_size)

        save_result = os.path.join(self.save_path, 'Results.csv')
        try:
            count = len(open(save_result, 'rU').readlines())
        except:
            count = 1

        net_weight = os.path.join(self.save_path, 'Weight')
        if not os.path.exists(net_weight):
            os.makedirs(net_weight)

        net_path = os.path.join(net_weight, str(count) + '.pkl')
        net_para_path = os.path.join(net_weight, str(count) + '_parameters.pkl')

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Let's use GPU: {}".format(self.device))
        else:
            print("Let's use CPU")

        # 实例化网络
        self.net = conv_bn_net(conv, flatten)
        if self.use_more_gpu and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs")
            # dim = 0 [64, xxx] -> [32, ...], [32, ...] on 2GPUs
            self.net = nn.DataParallel(self.net)
        self.net.to(device)
        try:
            self.net.load_state_dict(torch.load(net_para_path))
        except:
            pass

        self.net.train()
        try:
            optim = torch.optim.Adam(self.net.parameters(), lr=self.lr[0], weight_decay=self.weight_decay)
        except:
            optim = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = torch.nn.MSELoss()

        start = time.time()
        for e in range(self.epoch):

            # 是否使用 梯度衰减
            try:
                if (e + 1) % (self.epoch // 3) == 0:
                    optim.param_groups[0]["lr"] = self.lr[1]
                if (e + 1) % (self.epoch // 3 * 2) == 0:
                    optim.param_groups[0]["lr"] = self.lr[2]
            except:
                pass

            for b_train in batch_train_set:
                if torch.cuda.is_available():
                    #self.net = self.net.cuda()
                    train_x = Variable(b_train[:, :-1].view(-1, 1, self.W, self.H)).to(device)
                    train_y = Variable(b_train[:, -1:]).to(device)
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
                print('Training... epoch: {}, loss: {}'.format((e + 1), loss.cpu().data.numpy()))
                torch.save(self.net, net_path)
                torch.save(self.net.state_dict(), net_para_path)

                self.net.eval()
                if torch.cuda.is_available():
                    test_x = Variable(torch.FloatTensor(X_test).view(-1, 1, self.W, self.H)).to(device)
                    test_y = Variable(torch.FloatTensor(y_test)).to(device)
                else:
                    test_x = Variable(torch.FloatTensor(X_test).view(-1, 1, self.W, self.H))
                    test_y = Variable(torch.FloatTensor(y_test))

                test_prediction = self.net(test_x)
                test_loss = criterion(test_prediction, test_y)
                self.TestLosses.append(test_loss.cpu().data.numpy())

                self.test_prediction = test_prediction.cpu().data.numpy()
                self.test_prediction[self.test_prediction < 0] = 0

                print(
                    '\033[1;35m Testing... epoch: {}, loss: {} \033[0m!'.format((e + 1), test_loss.cpu().data.numpy()))
        end = time.time()
        self.t = end - start
        print('Training completed!!! Time consuming: {}'.format(str(self.t)))

    def predict(self, X_test):

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        self.net.eval()
        if torch.cuda.is_available():
            test_x = Variable(torch.FloatTensor(X_test).view(-1, 1, self.W, self.H)).to(device)
            prediction = self.net(test_x).to(device)
        else:
            test_x = Variable(torch.FloatTensor(X_test).view(-1, 1, self.W, self.H))
            prediction = self.net(test_x)

        prediction = prediction.cpu().data.numpy()
        prediction[prediction < 0] = 0
        return prediction

    def score(self):

        assert self.test_prediction is not None, '使用 score 前需要 fit'

        self.mse, self.rmse, self.mae, self.mape, \
        self.r2, self.r2_adjusted, self.rmsle = tools.reg_calculate(self.y_test, self.test_prediction,
                                                                    features=self.X_test.shape[-1])

    def __result_plot(self, save_file):

        plt.plot(range(len(self.test_prediction)), self.test_prediction, 'r--', label='prediction')
        plt.plot(range(len(self.y_test)), self.y_test, 'b--', label="true")
        plt.legend()
        try:
            plt.savefig(save_file)
            print('Save the picture successfully!')
        except:
            print('You have not define the path of saving!')
        plt.close()

    def __loss_plot(self, train_save, test_save):

        plt.plot(range(len(self.TrainLosses)), self.TrainLosses, 'r', label="train_loss")
        plt.legend()
        try:
            plt.savefig(train_save)
            print('Save the picture of training loss successfully!')
        except:
            print('You have not define the path of saving!')
        plt.close()

        plt.plot(range(len(self.TestLosses)), self.TestLosses, 'r', label="test_loss")
        plt.legend()
        try:
            plt.savefig(test_save)
            print('Save the picture of testing loss  successfully!')
        except:
            print('You have not define the path of saving!')

        plt.close()

    def __save_result(self, save_path):

        assert self.mse is not None, '需要先调用 score 函数'

        layer_numbers = len(self.channel_numbers)
        hidden_layers = str(self.channel_numbers).replace(',', '')
        try:
            lr = str(self.lr).replace(',', '')
        except:
            lr = self.lr
        count = tools.save_cnn_results(self.epoch, self.batch_size, lr, self.dropout, layer_numbers, hidden_layers,
                               self.kernel_size, self.conv_stride, self.pooling_size, self.pool_stride, self.flatten,
                               self.activate_function, self.weight_decay, self.mse, self.rmse, self.mae, self.mape, self.r2,
                               self.r2_adjusted, self.rmsle, self.is_standard, self.t,
                               self.Dimensionality_reduction_method, save_path, train_type='regression')
        print('Save results success!')
        return count

    def save(self):
        '''
        保存 pic， prediction，
        :return:
        '''

        # 保存 result，需要获得一个与行对应的数（count）
        result_path = os.path.join(self.save_path, 'Results.csv')
        count = self.__save_result(result_path)

        # 保存 prediction
        pred_path = os.path.join(self.save_path, 'Prediction')
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
        save_prediction = os.path.join(pred_path, str(count) + '.csv')
        np.savetxt(save_prediction, self.test_prediction, delimiter=',')
        print('Save the value of prediction successfully!!')

        # 保存图片
        pic_path = os.path.join(self.save_path, 'Pictures')
        if not os.path.exists(pic_path):
            os.makedirs(pic_path)
        pic_file = os.path.join(pic_path, str(count) + '.png')
        self.__result_plot(save_file=pic_file)

        # 保存 loss
        train_loss_path_pic = self.save_path + '/Loss/Train/pic'
        test_loss_path_pic = self.save_path + '/Loss/Test/pic'
        train_loss_path_value = self.save_path + '/Loss/Train/Values'
        test_loss_path_value = self.save_path + '/Loss/Test/Values'

        if not os.path.exists(train_loss_path_pic):
            os.makedirs(train_loss_path_pic)

        if not os.path.exists(test_loss_path_pic):
            os.makedirs(test_loss_path_pic)

        if not os.path.exists(train_loss_path_value):
            os.makedirs(train_loss_path_value)

        if not os.path.exists(test_loss_path_value):
            os.makedirs(test_loss_path_value)

        train_loss = os.path.join(train_loss_path_pic, 'Train' + str(count) + '.png')
        test_loss = os.path.join(test_loss_path_pic, 'Test' + str(count) + '.png')
        self.__loss_plot(train_save=train_loss, test_save=test_loss)

        train_loss_value = np.array(self.TrainLosses)
        test_loss_value = np.array(self.TestLosses)

        np.savetxt(train_loss_path_value + '/train' + str(count) + '.csv', train_loss_value, delimiter=',')
        np.savetxt(test_loss_path_value + '/test' + str(count) + '.csv', test_loss_value, delimiter=',')


'''
LSTM model
'''


class LSTM():
    def __init__(self, learning_rate, num_layers=2, hidden_size=32, dropout=0, activate_function='relu',
                 weight_decay=1e-8, device=0,
                 use_more_gpu=False, epoch=2000, batch_size=128, save_path='LSTM_Results',
                 is_standard=False, Dimensionality_reduction_method='None'):

        self.save_path = save_path  # 设置一条保存路径，直接把所有的值都收藏起来
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # self.layers = layers
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.dropout = dropout
        self.activate_function = activate_function
        self.weight_decay = weight_decay
        self.epoch = epoch
        self.batch_size = batch_size

        # 使用第几张 GPU,默认第 0 张，使用单卡。
        self.device = 'cuda:' + str(device)
        # 是否使用多卡
        self.use_more_gpu = use_more_gpu

        self.is_standard = is_standard
        self.Dimensionality_reduction_method = Dimensionality_reduction_method

        self.TrainLosses = []
        self.TestLosses = []

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

    def fit(self, X_train, y_train, X_test, y_test):
        '''
        :param X_train:
        :param y_train:
        :return:
        '''
        # if y is a scalar
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)


        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        input_size, output_size = X_train.shape[-1], y_train.shape[-1]

        b_data, b_labels = self.create_batch_size(X_train, y_train)

        # 搭建 lstm 网络
        # 判断 激活函数
        if self.activate_function == 'relu':
            activate_function = nn.ReLU()
        if self.activate_function == 'sigmoid':
            activate_function = nn.Sigmoid()
        if self.activate_function == 'tanh':
            activate_function = nn.Tanh()
        if self.activate_function == 'LeakyReLU':
            activate_function = nn.LeakyReLU()

        save_result = os.path.join(self.save_path, 'Results.csv')
        try:
            count = len(open(save_result, 'rU').readlines())
        except:
            count = 1

        net_weight = os.path.join(self.save_path, 'Weight')
        if not os.path.exists(net_weight):
            os.makedirs(net_weight)

        net_path = os.path.join(net_weight, str(count) + '.pkl')
        net_para_path = os.path.join(net_weight, str(count) + '_parameters.pkl')

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Let's use GPU: {}".format(self.device))
        else:
            print("Let's use CPU")
            
        # 实例化网络
        self.net = lstm_network(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                output_size=output_size, dropout=self.dropout, activate_function=activate_function)
        if self.use_more_gpu and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs")
            # dim = 0 [64, xxx] -> [32, ...], [32, ...] on 2GPUs
            self.net = nn.DataParallel(self.net)
        self.net.to(device)
        try:
            self.net.load_state_dict(torch.load(net_para_path))
        except:
            pass

        self.net.train()
        try:
            optim = torch.optim.Adam(self.net.parameters(), lr=self.lr[0], weight_decay=self.weight_decay)
        except:
            optim = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = torch.nn.MSELoss()

        start = time.time()
        for e in range(self.epoch):

            # 是否使用 梯度衰减
            self.net.train()
            try:
                if (e + 1) % (self.epoch // 3 * 2) == 0:
                    optim.param_groups[0]["lr"] = self.lr[1]
                if (e + 1) % (self.epoch // 3 * 2) == 0:
                    optim.param_groups[0]["lr"] = self.lr[2]
            except:
                pass

            for i in range(len(b_data)):
                if torch.cuda.is_available():
                    #print('cuda')
                    #self.net = self.net.cuda()
                    train_x = Variable(torch.FloatTensor(b_data[i])).to(device)
                    train_y = Variable(torch.FloatTensor(b_labels[i])).to(device)
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
                torch.save(self.net, net_path)
                torch.save(self.net.state_dict(), net_para_path)

                self.net.eval()
                if torch.cuda.is_available():
                    test_x = Variable(torch.FloatTensor(X_test)).to(device)
                    test_y = Variable(torch.FloatTensor(y_test)).to(device)
                else:
                    test_x = Variable(torch.FloatTensor(X_test))
                    test_y = Variable(torch.FloatTensor(y_test))

                test_prediction = self.net(test_x)
                test_loss = criterion(test_prediction, test_y)
                self.TestLosses.append(test_loss.cpu().data.numpy())

                self.test_prediction = test_prediction.cpu().data.numpy()
                self.test_prediction[self.test_prediction < 0] = 0

                print(
                    '\033[1;35m Testing... epoch: {}, loss: {} \033[0m!'.format((e + 1), test_loss.cpu().data.numpy()))

        end = time.time()
        self.t = end - start
        print('Training completed!!! Time consuming: {}'.format(str(self.t)))

    def predict(self, X_test):

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        self.net.eval()
        if torch.cuda.is_available():
            test_x = Variable(torch.FloatTensor(X_test)).to(device)
        else:
            test_x = Variable(torch.FloatTensor(X_test))

        prediction = self.net(test_x)
        prediction = prediction.cpu().data.numpy()
        prediction[prediction < 0] = 0
        return prediction

    def score(self):

        assert self.test_prediction is not None, '使用 score 前需要 fit'

        self.mse, self.rmse, self.mae, self.mape, \
        self.r2, self.r2_adjusted, self.rmsle = tools.reg_calculate(self.y_test,
                                                                    self.test_prediction, self.X_test.shape[-1])

    def __result_plot(self, save_file):

        plt.plot(range(len(self.test_prediction)), self.test_prediction, 'r--', label='prediction')
        plt.plot(range(len(self.y_test)), self.y_test, 'b--', label="true")
        plt.legend()
        try:
            plt.savefig(save_file)
            print('Save the picture successfully!')
        except:
            print('You have not define the path of saving!')
        plt.close()

    def __loss_plot(self, train_save, test_save):

        plt.plot(range(len(self.TrainLosses)), self.TrainLosses, 'r', label="train_loss")
        plt.legend()
        try:
            plt.savefig(train_save)
            print('Save the picture of training loss successfully!')
        except:
            print('You have not define the path of saving!')
        plt.close()

        plt.plot(range(len(self.TestLosses)), self.TestLosses, 'r', label="test_loss")
        plt.legend()
        try:
            plt.savefig(test_save)
            print('Save the picture of testing loss  successfully!')
        except:
            print('You have not define the path of saving!')

        plt.close()

    def __save_result(self, save_path):

        assert self.mse is not None, '需要先调用 score 函数'

        try:
            lr = str(self.lr).replace(',', '')
        except:
            lr = self.lr
        count = tools.save_lstm_results(self.epoch, self.batch_size, lr, self.dropout, self.num_layers, self.hidden_size,
                                self.activate_function, self.weight_decay, self.mse, self.rmse, self.mae, self.mape, self.r2,
                                self.r2_adjusted, self.rmsle,
                                self.is_standard, self.Dimensionality_reduction_method, self.t,
                                save_path, train_type='regression')
        print('Save results success!')
        return count

    def save(self):
        '''
        保存 pic， prediction，
        :return:
        '''

        # 保存 result，需要获得一个与行对应的数（count）
        result_path = os.path.join(self.save_path, 'Results.csv')
        count = self.__save_result(result_path)

        # 保存 prediction
        pred_path = os.path.join(self.save_path, 'Prediction')
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
        save_prediction = os.path.join(pred_path, str(count) + '.csv')
        np.savetxt(save_prediction, self.test_prediction, delimiter=',')
        print('Save the value of prediction successfully!!')

        # 保存图片
        pic_path = os.path.join(self.save_path, 'Pictures')
        if not os.path.exists(pic_path):
            os.makedirs(pic_path)
        pic_file = os.path.join(pic_path, str(count) + '.png')
        self.__result_plot(save_file=pic_file)

        # 保存 loss
        train_loss_path_pic = self.save_path + '/Loss/Train/pic'
        test_loss_path_pic = self.save_path + '/Loss/Test/pic'
        train_loss_path_value = self.save_path + '/Loss/Train/Values'
        test_loss_path_value = self.save_path + '/Loss/Test/Values'

        if not os.path.exists(train_loss_path_pic):
            os.makedirs(train_loss_path_pic)

        if not os.path.exists(test_loss_path_pic):
            os.makedirs(test_loss_path_pic)

        if not os.path.exists(train_loss_path_value):
            os.makedirs(train_loss_path_value)

        if not os.path.exists(test_loss_path_value):
            os.makedirs(test_loss_path_value)

        train_loss = os.path.join(train_loss_path_pic, 'Train' + str(count) + '.png')
        test_loss = os.path.join(test_loss_path_pic, 'Test' + str(count) + '.png')
        self.__loss_plot(train_save=train_loss, test_save=test_loss)

        train_loss_value = np.array(self.TrainLosses)
        test_loss_value = np.array(self.TestLosses)

        np.savetxt(train_loss_path_value + '/train' + str(count) + '.csv', train_loss_value, delimiter=',')
        np.savetxt(test_loss_path_value + '/test' + str(count) + '.csv', test_loss_value, delimiter=',')
