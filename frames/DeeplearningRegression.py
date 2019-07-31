import torch
from torch.autograd import Variable
from torch import nn
import matplotlib.pyplot as plt
from . import tools


class ANN():
    def __init__(self, hidden_layers, learning_rate=0.01, activate_function='relu', epoch=2000, batch_size=128):
        # self.layers = layers
        self.hidden_layers = hidden_layers
        self.lr = learning_rate
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

        # hidden layers
        hidden_layers_number = len(hidden_layers)  # 隐藏层个数

        for i in range(hidden_layers_number):
            try:
                layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
                layers.append(activate_function)
            except:
                pass

        # output layer
        output_layer = nn.Linear(hidden_layers[-1], output_size)
        layers.append(output_layer)

        # create network 找不到好办法，只能用 if else 去判断网络层数来搭建，这样的缺陷是：网络不能动态调整
        if hidden_layers_number == 1:
            seq_net = nn.Sequential(
                layers[0], layers[1], layers[2]
            )

        if hidden_layers_number == 2:
            seq_net = nn.Sequential(
                layers[0], layers[1], layers[2], layers[3], layers[4]
            )
        if hidden_layers_number == 3:
            seq_net = nn.Sequential(
                layers[0], layers[1], layers[2], layers[3], layers[4], layers[5], layers[6]
            )

        if hidden_layers_number == 4:
            seq_net = nn.Sequential(
                layers[0], layers[1], layers[2], layers[3], layers[4], layers[5], layers[6], layers[7], layers[8]
            )

        if hidden_layers_number == 5:
            seq_net = nn.Sequential(
                layers[0], layers[1], layers[2], layers[3], layers[4], layers[5], layers[6], layers[7], layers[8], layers[9], layers[10]
            )

        return seq_net


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

        self.net = self.model(input_size, output_size)
        self.net.train()
        optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = torch.nn.MSELoss()

        for e in range(self.epoch):
            if torch.cuda.is_available():
                #print('cuda')
                self.net = self.net.cuda()
                train_x = Variable(torch.FloatTensor(X_train)).cuda()
                train_y = Variable(torch.FloatTensor(y_train)).cuda()
            else:
                train_x = Variable(torch.FloatTensor(X_train))
                train_y = Variable(torch.FloatTensor(y_train))

            prediction = self.net(train_x)

            loss = criterion(prediction, train_y)
            self.TrainLosses.append(loss.cpu().data.numpy())

            optim.zero_grad()
            loss.backward()
            optim.step()

            if (e + 1) % 100 == 0:
                print('Training... epoch: {}, loss: {}'.format((e+1), loss.cpu().data.numpy()))

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

    def loss_plot(self):

        plt.plot(range(len(self.TrainLosses)), self.TrainLosses, 'r', label="loss")
        plt.legend()
        plt.show()

    def save_result(self, save_path, is_standard=False, is_PCA=False):
        layer_numbers = len(self.hidden_layers)
        hidden_layers = str(self.hidden_layers).replace(',', '')
        tools.save_results(self.epoch, self.lr, layer_numbers, hidden_layers,
                           self.mse, self.rmse, self.mae, self.r2, is_standard, is_PCA, save_path)
        print('Save results success!')