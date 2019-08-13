import numpy as np
from . import tools
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns


'''
    ELM model (极限学习机)
'''


class ELMRegression:
    def __init__(self, hidden_nodes):
        # num_hidden 隐藏层神经元个数
        self.hidden_nodes = hidden_nodes

    # activate function
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    # 开始训练
    def fit(self, X_train, y_train):
        '''
        该函数重点是求出 输出权重 out_w
        '''
        self.x_train = np.atleast_2d(X_train)  # 强转成 2 维
        self.y_train = y_train.flatten()  # 变成 标量
        self.train_sample_numbers = len(self.x_train)  # sample 个数
        self.feature_numbers = self.x_train.shape[1]

        # 随机生成 W （-1，1）
        self.W = np.random.uniform(-1, 1, (self.feature_numbers, self.hidden_nodes))
        # 随机生成 b
        b = np.random.uniform(-0.6, 0.6, (1, self.hidden_nodes))
        self.first_b = b

        # 生成偏置矩阵
        # 生成都是 b数据的 二维矩阵 shape = x_train
        for i in range(self.train_sample_numbers - 1):
            b = np.row_stack((b, self.first_b))

        self.b = b

        # 神经元输出值 H
        H = self.sigmoid(np.dot(self.x_train, self.W) + self.b)
        # H 的广义逆矩阵
        H_ = np.linalg.pinv(H)  # shape 反过来

        self.out_w = np.dot(H_, self.y_train)  # 输出权重

    def predict(self, x_test):
        assert self.out_w is not None, \
            "must fit before predict"

        self.x_test = np.atleast_2d(x_test)
        self.test_sample_numbers = len(self.x_test)
        self.pre_Y = np.zeros((x_test.shape[0]))

        b = self.first_b

        # 扩充偏置矩阵，跟初始化那里一致
        for i in range(self.test_sample_numbers - 1):
            b = np.row_stack((b, self.first_b))

        # predict
        self.pre_Y = np.dot(self.sigmoid(np.dot(self.x_test, self.W) + b), self.out_w)
        #         prediction = np.argmax(self.pre_Y, axis=1)
        prediction = self.pre_Y

        return prediction

    def score(self, x_test, y_test):
        prediction = self.predict(x_test)
        self.mse, self.rmse, self.mae, self.r2 = tools.reg_calculate(y_test, prediction)
        return self.mse, self.rmse, self.mae, self.r2

    def result_plot(self, X_test, y_test, save_file, is_show=False):

        prediction = self.predict(X_test)
        plt.plot(range(len(prediction)), prediction, 'r--', label='prediction')
        plt.plot(range(len(y_test)), y_test, 'b--', label="true")
        plt.legend()
        plt.savefig(save_file)
        if is_show:
            plt.show()
        plt.close()
        print('Save the picture successfully!')

    def save_result(self, save_path, is_standard=False, Dimensionality_reduction_method='None'):
        tools.save_elm(self.hidden_nodes, self.mse, self.rmse, self.mae, self.r2, is_standard,
                       Dimensionality_reduction_method, save_file=save_path, train_type='Regression')
        print('Save results success!')

    def __repr__(self):
        return "EML(hidden_neurons:%d" % self.hidden_nodes

'''
Classicifation
'''


class ELMClassification(object):
    def __init__(self, hidden_nodes):
        # num_hidden 隐藏层神经元个数
        self.hidden_nodes = hidden_nodes
        self.onehot = OneHotEncoder(sparse=False)

    # activate function
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    # 开始训练
    def fit(self, X_train, y_train):
        '''
        该函数重点是求出 输出权重 out_w
        '''
        self.x_train = np.atleast_2d(X_train)  # 强转成 2 维
        self.y_train = y_train.flatten()  # 变成 标量
        self.train_sample_numbers = len(self.x_train)  # sample 个数
        self.feature_numbers = self.x_train.shape[1]

        # 随机生成 W （-1，1）
        self.W = np.random.uniform(-1, 1, (self.feature_numbers, self.hidden_nodes))
        # 随机生成 b
        b = np.random.uniform(-0.6, 0.6, (1, self.hidden_nodes))
        self.first_b = b

        # 生成偏置矩阵
        # 生成都是 b数据的 二维矩阵 shape = x_train
        for i in range(self.train_sample_numbers - 1):
            b = np.row_stack((b, self.first_b))

        self.b = b

        # 神经元输出值 H
        H = self.sigmoid(np.dot(self.x_train, self.W) + self.b)
        # H 的广义逆矩阵
        H_ = np.linalg.pinv(H)  # shape 反过来

        # 将 y 转成 one-hot 编码
        # self.train_y = np.zeros((self.train_sample_numbers, classes))
        self.train_y = self.onehot.fit_transform(y_train.reshape(-1, 1))
        self.output_size = self.train_y.shape[1]
        for i in range(0, self.train_sample_numbers):
            self.train_y[i, y_train[i]] = 1

        self.out_w = np.dot(H_, self.train_y)  # 输出权重

    def predict(self, x_test):
        assert self.out_w is not None, \
            "must fit before predict"

        self.x_test = np.atleast_2d(x_test)
        self.test_sample_numbers = len(self.x_test)
        self.pre_Y = np.zeros((x_test.shape[0]))

        b = self.first_b

        # 扩充偏置矩阵，跟初始化那里一致
        for i in range(self.test_sample_numbers - 1):
            b = np.row_stack((b, self.first_b))

        # predict
        self.pre_Y = np.dot(self.sigmoid(np.dot(self.x_test, self.W) + b), self.out_w)
        prediction = np.argmax(self.pre_Y, axis=1)

        return prediction

    def score(self, x_test, y_test):
        prediction = self.predict(x_test)
        self.acc, self.precision, self.recall, self.f1 = tools.clf_calculate(y_test, prediction)
        return self.acc, self.precision, self.recall, self.f1

    def confusion_matrix_result(self, X_test, y_test, save_file, is_show=False, delete_zero=False):
        ''' delete_zero, 是否除掉 0 标签，适用于极端数据 '''
        x, y = [], []
        prediction = self.predict(X_test)
        ann = tools.confusion_matrix_result(y_test, prediction)
        if delete_zero:
            ann = ann[1:, 1:]
            for i in range(1, self.output_size):
                x.append('P' + str(i))
                y.append('T' + str(i))
        else:
            for i in range(self.output_size):
                x.append('P' + str(i))
                y.append('T' + str(i))

        pic = sns.heatmap(ann, annot=True, yticklabels=y, linewidths=0.5, xticklabels=x, cmap="YlGnBu")
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        sns.set(font_scale=1.5)
        pic_save = pic.get_figure()
        pic_save.savefig(save_file)
        if is_show:
            plt.show()
        plt.close()
        print('Save the picture successfully!')

    def save_result(self, save_path, is_standard=False, Dimensionality_reduction_method='None'):
        tools.save_elm(self.hidden_nodes, self.acc, self.precision, self.recall, self.f1, is_standard,
                       Dimensionality_reduction_method, save_file=save_path, train_type='Classification')
        print('Save results success!')

    def __repr__(self):
        return "EML(hidden_neurons:%d" % self.hidden_nodes