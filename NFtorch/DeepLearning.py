import torch
from torch.autograd import Variable
from torch import nn

import numpy as np
import matplotlib.pyplot as plt
from NFtorch import tools

import os
import time

import warnings
warnings.filterwarnings("ignore")


class Regression:
    def __init__(self, model,
                 learning_rate,
                 dropout=0,
                 epoch=2000,
                 batch_size=128,
                 weight_decay=1e-8):

        self.model = model
        self.lr = learning_rate
        self.dropout = dropout
        self.epoch = epoch
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        self.TrainLosses = []
        self.TestLosses = []
        self.t = 0
        self.D = []
        self.n = 0  # 来记录 梯度衰减 的次数
        self.limit = [1e-2, 1e-3, 1e-4]

        self.device = 0
        self.use_more_gpu = False
        self.preprocessing_method = 'None'
        self.decomposition_method = 'None'
        self.save_path = 'regression_save'

        self.mse = -1
        self.rmse = -1
        self.mae = -1
        self.mad = -1
        self.mape = -1
        self.r2 = -1
        self.r2_adjusted = -1
        self.rmsle = -1

    def set_device(self, device=0, use_more_gpu=False):
        '''
        機器 GPU 設置，默認使用單卡 0.
        :param device:
        :param use_more_gpu:
        :return:
        '''
        self.device = device
        self.use_more_gpu = use_more_gpu

    def set_methods(self, preprocessing_method='None', decomposition_method='None', save_path='regression_save'):
        '''
        存儲數據與處理操作到 csv，以及保存路徑。
        :param preprocessing_method:
        :param decomposition_method:
        :param save_path:
        :return:
        '''
        self.preprocessing_method = preprocessing_method
        self.decomposition_method = decomposition_method

        self.save_path = save_path  # 设置一条保存路径，直接把所有的值都收藏起来

    def set_limit(self, limit):
        '''
        设置中止装值的limit
        :param limit:
        :return:
        '''
        self.limit = limit

    def create_batch_size(self, X_train, y_train):
        '''
        構造 mini batch 數據集
        :param X_train:
        :param y_train:
        :return:
        '''
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
        訓練模型，傳入訓練集和測試集。
        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :return:
        '''

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        ''' training the network '''
        # if y is a scalar
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)


        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        b_datas, b_labels = self.create_batch_size(X_train, y_train)

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


        if torch.cuda.is_available():
            if self.use_more_gpu and torch.cuda.device_count() > 1:
                print("Now we're using ", torch.cuda.device_count(), " GPUs")
                # dim = 0 [64, xxx] -> [32, ...], [32, ...] on 2GPUs
                self.model = nn.DataParallel(self.model)
            else:
                print("Now we're using the GPU : {}".format(self.device))
        else:
            print("Now we're using the CPU")

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        self.model.to(device)

        try:
            self.model.load_state_dict(torch.load(net_para_path))
        except:
            pass

        self.model.train()
        try:
            optim = torch.optim.Adam(self.model.parameters(), lr=self.lr[0], weight_decay=self.weight_decay)
        except:
            optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = torch.nn.MSELoss()

        start = time.time() # 计算时间
        limit = self.limit[0]
        for e in range(self.epoch):

            for i in range(len(b_datas)):
                if torch.cuda.is_available():
                    train_x = Variable(torch.FloatTensor(b_datas[i])).to(device)
                    train_y = Variable(torch.FloatTensor(b_labels[i])).to(device)
                else:
                    train_x = Variable(torch.FloatTensor(b_datas[i]))
                    train_y = Variable(torch.FloatTensor(b_labels[i]))

                prediction = self.model(train_x)

                loss = criterion(prediction, train_y)

                optim.zero_grad()
                loss.backward()
                optim.step()

            self.TrainLosses.append(loss.cpu().data.numpy())  # 收集 训练集 的 loss

            self.D.append(loss.cpu().data.numpy())

            # epoch 终止装置
            if len(self.D) >= 20:
                loss1 = np.mean(np.array(self.D[-20:-10]))
                loss2 = np.mean(np.array(self.D[-10:]))
                d = np.float(np.abs(loss2 - loss1))

                if d < limit or e == self.epoch-1:  # 加入遍历完都没达成limit限定，就直接得到结果
                    self.D = []  # 重置
                    self.n += 1
                    print('The error changes within {}'.format(limit))
                    self.e = e + 1

                    print(
                        'Training... epoch: {}, loss: {}'.format((e + 1), loss.cpu().data.numpy()))
                    torch.save(self.model, net_path)
                    torch.save(self.model.state_dict(), net_para_path)

                    self.model.eval()
                    if torch.cuda.is_available():
                        test_x = Variable(torch.FloatTensor(self.X_test)).to(device)
                        test_y = Variable(torch.FloatTensor(self.y_test)).to(device)
                    else:
                        test_x = Variable(torch.FloatTensor(self.X_test))
                        test_y = Variable(torch.FloatTensor(self.y_test))

                    test_prediction = self.model(test_x)
                    test_loss = criterion(test_prediction, test_y)
                    self.TestLosses.append(test_loss.cpu().data.numpy())

                    self.test_prediction = test_prediction.cpu().data.numpy()  # 收集测试的 loss
                    self.test_prediction[self.test_prediction < 0] = 0

                    print('\033[1;35m Testing... epoch: {}, loss: {} \033[0m!'.format((e + 1),
                                                                                      test_loss.cpu().data.numpy()))

                    # 已经梯度衰减了 2 次
                    if self.n == 3:
                        print('The meaning of the loop is not big, stop!!')
                        break
                    # IF learning rate is not list，break.
                    if type(self.lr) is not list:
                        break

                    try:
                        # 沒有給足三個 learning rate
                        limit = self.limit[self.n]
                        print('Now learning rate is : {}'.format(self.lr[self.n]))
                        optim.param_groups[0]["lr"] = self.lr[self.n]
                    except:
                        print('You only give one learning rate!!!')
                        break

        end = time.time()
        self.t = end - start
        print('Training completed!!! Time consuming: {}'.format(str(self.t)))

        # 计算结果
        self.mse, self.rmse, self.mae, self.mad, self.mape, \
        self.r2, self.r2_adjusted, self.rmsle = tools.reg_calculate(self.y_test, self.test_prediction,
                                                                    self.X_test.shape[-1])

    def predict(self, X_test):
        '''
        預測結果
        :param X_test:
        :return:
        '''
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        self.model.eval()
        if torch.cuda.is_available():
            test_x = Variable(torch.FloatTensor(X_test)).to(device)
        else:
            test_x = Variable(torch.FloatTensor(X_test))

        prediction = self.model(test_x)

        prediction = prediction.cpu().data.numpy()
        prediction[prediction < 0] = 0

        return prediction

    def score(self):
        '''
        得到結果
        :return:
        '''
        assert self.test_prediction is not None, 'Need to "fit" before "socre"'

        print("mse: {}, rmse: {}, mae: {}, mape: {}, "
              "r2: {}, r2_adjusted: {}, rmsle: {}".format(self.mse, self.rmse, self.mae, self.mape,
                                                          self.r2, self.r2_adjusted, self.rmsle))

    def __result_plot(self, save_file):
        '''
        保存預測結果圖
        :param save_file:
        :return:
        '''
        color_sequence = ['#1f77b4', '#aec7e8']

        plt.figure(num=None, figsize=(8, 2), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(range(len(self.test_prediction)), self.test_prediction,
                 '-o', label='prediction', color=color_sequence[0], linewidth=2)
        plt.plot(range(len(self.y_test)), self.y_test, '--o', label='y_true', color=color_sequence[1], linewidth=2)
        plt.grid()
        plt.legend()
        try:
            plt.savefig(save_file, dpi=1000, bbox_inches='tight')
            print('Save the picture successfully!')
        except:
            print('You have not define the path of saving!')
        plt.close()

    def __loss_plot(self, train_save, test_save):
        '''
        保存 training losses 和 testing losses 的結果圖
        :param train_save:
        :param test_save:
        :return:
        '''
        # 保存 training losses
        plt.plot(range(len(self.TrainLosses)), self.TrainLosses, 'r', label="train_loss")
        plt.legend()
        plt.savefig(train_save)
        plt.close()

        # 保存 testing losses
        plt.plot(range(len(self.TestLosses)), self.TestLosses, 'r', label="test_loss")
        plt.legend()
        plt.savefig(test_save)
        plt.close()

    def __save_result(self, save_path):

        try:
            lr = str(self.lr).replace(',', '')
        except:
            lr = self.lr
        count = tools.save_dl_regression(self.e, self.batch_size, lr, self.dropout,
                                         self.weight_decay,
                                         self.mse, self.rmse,
                                         self.mae, self.mad, self.mape, self.r2, self.r2_adjusted, self.rmsle,
                                         self.preprocessing_method,
                                         self.decomposition_method, t=self.t,
                                         save_result=save_path)
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


class Classification:
    '''
    分类模型
    '''
    def __init__(self, model,
                 learning_rate,
                 dropout=0,
                 epoch=2000,
                 batch_size=128,
                 weight_decay=1e-8):

        self.model = model
        self.lr = learning_rate
        self.dropout = dropout
        self.epoch = epoch
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        self.TrainLosses = []
        self.TestLosses = []
        self.t = 0
        self.D = []
        self.n = 0  # 来记录 梯度衰减 的次数
        self.limit = [1e-2, 1e-3, 1e-4]

        self.device = 0
        self.use_more_gpu = False
        self.preprocessing_method = 'None'
        self.decomposition_method = 'None'
        self.save_path = 'classification_save'

        self.acc = -1
        self.precision = -1
        self.recall = -1
        self.f1 = -1

    def set_device(self, device=0, use_more_gpu=False):
        '''
        機器 GPU 設置，默認使用單卡 0.
        :param device:
        :param use_more_gpu:
        :return:
        '''
        self.device = device
        self.use_more_gpu = use_more_gpu

    def set_methods(self, preprocessing_method='None', decomposition_method='None', save_path='classification_save'):
        '''
        存儲數據與處理操作到 csv，以及保存路徑。
        :param preprocessing_method:
        :param decomposition_method:
        :param save_path:
        :return:
        '''
        self.preprocessing_method = preprocessing_method
        self.decomposition_method = decomposition_method

        self.save_path = save_path  # 设置一条保存路径，直接把所有的值都收藏起来

    def set_limit(self, limit):
        '''
        设置中止装值的limit
        :param limit:
        :return:
        '''
        self.limit = limit

    def create_batch_size(self, X_train, y_train):
        '''
        構造 mini batch 數據集
        :param X_train:
        :param y_train:
        :return:
        '''
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

    def __get_acc(self, prediction, true):
        ''' calculate the accurary '''
        prediction = prediction.cpu()
        _, pred = torch.max(prediction, 1)
        correct_number = torch.sum(pred == true)

        acc = correct_number.numpy() / len(true)
        return acc

    def fit(self, X_train, y_train, X_test, y_test):
        '''
        訓練模型，傳入訓練集和測試集。
        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :return:
        '''

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        ''' training the network '''

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train.flatten(), \
                                                               y_test.flatten()

        b_datas, b_labels = self.create_batch_size(X_train, y_train)

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

        if torch.cuda.is_available():
            if self.use_more_gpu and torch.cuda.device_count() > 1:
                print("Now we're using ", torch.cuda.device_count(), " GPUs")
                # dim = 0 [64, xxx] -> [32, ...], [32, ...] on 2GPUs
                self.model = nn.DataParallel(self.model)
            else:
                print("Now we're using the GPU : {}".format(self.device))
        else:
            print("Now we're using the CPU")

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        self.model.to(device)

        try:
            self.model.load_state_dict(torch.load(net_para_path))
        except:
            pass

        self.model.train()
        try:
            optim = torch.optim.Adam(self.model.parameters(), lr=self.lr[0], weight_decay=self.weight_decay)
        except:
            optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        start = time.time() # 计算时间
        limit = self.limit[0]
        for e in range(self.epoch):

            for i in range(len(b_datas)):
                if torch.cuda.is_available():
                    train_x = Variable(torch.FloatTensor(b_datas[i])).to(device)
                    train_y = Variable(torch.FloatTensor(b_labels[i]).type(torch.int64)).to(device)
                else:
                    train_x = Variable(torch.FloatTensor(b_datas[i]))
                    train_y = Variable(torch.FloatTensor(b_labels[i]).type(torch.int64))

                prediction = self.model(train_x)
                loss = criterion(prediction, train_y)

                optim.zero_grad()
                loss.backward()
                optim.step()

            self.TrainLosses.append(loss.cpu().data.numpy())  # 收集 训练集 的 loss

            self.D.append(loss.cpu().data.numpy())

            # epoch 终止装置
            if len(self.D) >= 20:
                loss1 = np.mean(np.array(self.D[-20:-10]))
                loss2 = np.mean(np.array(self.D[-10:]))
                d = np.float(np.abs(loss2 - loss1))

                if d < limit or e == self.epoch-1:  # 加入遍历完都没达成limit限定，就直接得到结果
                    self.D = []  # 重置
                    self.n += 1
                    print('The error changes within {}'.format(limit))
                    self.e = e + 1

                    train_acc = self.__get_acc(prediction, train_y)
                    print(
                        'Training... epoch: {}, loss: {}, acc: {}'.format((e + 1), loss.cpu().data.numpy(), train_acc))
                    torch.save(self.model, net_path)
                    torch.save(self.model.state_dict(), net_para_path)

                    self.model.eval()
                    if torch.cuda.is_available():
                        test_x = Variable(torch.FloatTensor(self.X_test)).to(device)
                        test_y = Variable(torch.FloatTensor(self.y_test).type(torch.int64)).to(device)
                    else:
                        test_x = Variable(torch.FloatTensor(self.X_test))
                        test_y = Variable(torch.FloatTensor(self.y_test).type(torch.int64))

                    self.test_prediction = self.model(test_x)
                    test_loss = criterion(self.test_prediction, test_y)
                    self.TestLosses.append(test_loss.cpu().data.numpy())

                    test_acc = self.__get_acc(self.test_prediction, test_y)
                    print('\033[1;35m Testing... epoch: {}, loss: {}, '
                          'acc: {} \033[0m!'.format((e + 1), test_loss.cpu().data.numpy(), test_acc))

                    # 已经梯度衰减了 2 次
                    if self.n == 3:
                        print('The meaning of the loop is not big, stop!!')
                        break
                    # IF learning rate is not list，break.
                    if type(self.lr) is not list:
                        break

                    try:
                        # 沒有給足三個 learning rate
                        limit = self.limit[self.n]
                        print('Now learning rate is : {}'.format(self.lr[self.n]))
                        optim.param_groups[0]["lr"] = self.lr[self.n]
                    except:
                        print('You only give one learning rate!!!')
                        break

        end = time.time()
        self.t = end - start
        print('Training completed!!! Time consuming: {}'.format(str(self.t)))

        # 计算结果
        self.prediction = torch.argmax(self.test_prediction, dim=1).cpu().data.numpy()
        self.acc, self.precision, self.recall, self.f1 = tools.clf_calculate(self.y_test, self.prediction)

    def predict(self, X_test):
        '''
        預測結果
        :param X_test:
        :return:
        '''
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        self.net.eval()
        if torch.cuda.is_available():
            test_x = Variable(torch.FloatTensor(X_test)).to(device)
        else:
            test_x = Variable(torch.FloatTensor(X_test))

        prediction = self.net(test_x)
        prediction = torch.argmax(prediction, dim=1)
        return prediction.cpu().data.numpy()

    def score(self):
        '''
        得到結果
        :return:
        '''
        assert self.prediction is not None, 'Need to "fit" before "socre"'

        print('acc: {}, precision: {}, recall: {}, f1: {}'.format(self.acc, self.precision, self.recall, self.f1))

    def __loss_plot(self, train_save, test_save):
        '''
        保存 training losses 和 testing losses 的結果圖
        :param train_save:
        :param test_save:
        :return:
        '''
        # 保存 training losses
        plt.plot(range(len(self.TrainLosses)), self.TrainLosses, 'r', label="train_loss")
        plt.legend()
        plt.savefig(train_save)
        plt.close()

        # 保存 testing losses
        plt.plot(range(len(self.TestLosses)), self.TestLosses, 'r', label="test_loss")
        plt.legend()
        plt.savefig(test_save)
        plt.close()

    def __save_result(self, save_path):

        try:
            lr = str(self.lr).replace(',', '')
        except:
            lr = self.lr
        count = tools.save_dl_classification(self.e, self.batch_size, lr, self.dropout,
                                             self.weight_decay,
                                             self.acc, self.precision,
                                             self.recall, self.f1,
                                             self.preprocessing_method,
                                             self.decomposition_method, t=self.t,
                                             save_result=save_path)
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
        np.savetxt(save_prediction, self.prediction, delimiter=',')
        print('Save the value of prediction successfully!!')

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
