import numpy as np
from sklearn import metrics
import os
from sklearn.metrics import confusion_matrix
import torch
import csv

import matplotlib.pyplot as plt

'''
This is a toolkit that provides calculations, save results, create data set, plot result pictures.
'''


def reg_calculate(true, prediction, features=None):
    '''
        To calculate the result of regression,
        including mse, rmse, mae, r2, four criterions.
    '''
    prediction[prediction < 0] = 0

    mse = metrics.mean_squared_error(true, prediction)
    rmse = np.sqrt(mse)

    mae = metrics.mean_absolute_error(true, prediction)
    mad = metrics.median_absolute_error(true, prediction)
    mape = np.mean(np.abs((true - prediction) / true)) * 100

    r2 = metrics.r2_score(true, prediction)
    rmsle = np.sqrt(metrics.mean_squared_log_error(true, prediction))

    try:
        n = len(true)
        p = features
        r2_adjusted = 1-((1-metrics.r2_score(true, prediction))*(n-1))/(n-p-1)
    except:
        # print("mse: {}, rmse: {}, mae: {}, mape: {}, r2: {}, rmsle: {}".format(mse, rmse, mae, mape, r2, rmsle))
        # print('if you wanna get the value of r2_adjusted, you can define the number of features, '
        #       'which is the third parameter.')
        return mse, rmse, mae, mape, r2, rmsle

    # print("mse: {}, rmse: {}, mae: {}, mape: {}, r2: {}, r2_adjusted: {}, rmsle: {}".format(mse, rmse, mae, mape,
    #                                                                                         r2, r2_adjusted, rmsle))
    return mse, rmse, mae, mad, mape, r2, r2_adjusted, rmsle


def clf_calculate(true, prediction):
    '''
        To calculate the result of classification,
        including acc, precision, recall, f1, four criterions.
    '''
    acc = metrics.accuracy_score(true, prediction)
    precision = metrics.precision_score(true, prediction, average='macro')
    recall = metrics.recall_score(true, prediction, average='macro')
    f1 = metrics.f1_score(true, prediction, average='macro')

    #print('acc: {}, precision: {}, recall: {}, f1: {}'.format(acc, precision, recall, f1))
    return acc, precision, recall, f1


def confusion_matrix_result(true, prediction):
    '''
        Got the confusion matrix.
    '''
    cfm = confusion_matrix(true, prediction)
    return cfm


def create_dataset(dataset, look_back=7, need_label=True):
    '''
        Create new structure of origin dataset to adapt LSTM network.
    '''
    dataX, dataY = [], []

    if dataset.shape[1] == 1:
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back)]
            dataX.append(a)
            dataY.append(dataset[i + look_back])
    else:
        if need_label:
            for i in range(len(dataset) - look_back):
                a = dataset[i:(i + look_back)]
                dataX.append(a)
                dataY.append(dataset[i + look_back, -1:])
        else:
            for i in range(len(dataset) - look_back):
                a = dataset[i:(i + look_back), :-1]
                dataX.append(a)
                dataY.append(dataset[i + look_back, -1:])
    return np.array(dataX), np.array(dataY)


def train_test_split(X, y, test_size=0.2, random_state=19):

    np.random.seed(random_state)
    train_size = int(len(X) * (1 - test_size))
    p = np.random.permutation(X.shape[0])
    data = X[p]
    label = y[p]

    X_train, X_test = data[:train_size], data[train_size:]
    y_train, y_test = label[:train_size], label[train_size:]
    return X_train, X_test, y_train, y_test


def cross_entropy_erorr(y, t):
    ''' Self-defined cross entropy function '''
    delta = 1e-7
    return -torch.sum(t * torch.log(y + delta))


def save_dl_regression(epoch, batch_size, lr, dropout, weight_decay,
                       mse, rmse, mae, mad, mape, r2, r2_adjusted, rmsle,
                     preprocessing_method, decomposition_method, t, save_result):

    # 计算行数，匹配 prediciton 的保存
    try:
        count = len(open(save_result, 'rU').readlines())
    except:
        count = 1

    if not os.path.exists(save_result):
        content = ['Count', 'epoch', 'batch_size', 'lr', 'dropout', 'weight_decay',
                   'mse', 'rmse', 'mae', 'mad', 'mape', 'r2', 'r2_adjusted', 'rmsle',
                   'preprocessing_method', 'decomposition_method', 'Time']
        with open(save_result, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(content)

    content = [count, epoch, batch_size, lr, dropout, weight_decay,
               mse, rmse, mae, mad, mape, r2, r2_adjusted, rmsle,
               preprocessing_method, decomposition_method, t]
    with open(save_result, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(content)

    return count


def save_dl_classification(epoch, batch_size, lr, dropout, weight_decay,
                       acc, precision, recall, f1, preprocessing_method, decomposition_method, t, save_result):

    # 计算行数，匹配 prediciton 的保存
    try:
        count = len(open(save_result, 'rU').readlines())
    except:
        count = 1

    if not os.path.exists(save_result):
        content = ['Count', 'epoch', 'batch_size', 'lr', 'dropout', 'weight_decay',
                   'accuracy', 'precision', 'recall', 'f1',
                   'preprocessing_method', 'decomposition_method', 'Time']
        with open(save_result, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(content)

    content = [count, epoch, batch_size, lr, dropout, weight_decay,
               acc, precision, recall, f1,
               preprocessing_method, decomposition_method, t]
    with open(save_result, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(content)

    return count