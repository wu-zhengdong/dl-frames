# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 03:18:22 2018


@author: HAN_RUIZHI yb77447@umac.mo OR  501248792@qq.com

This code is the first version of BLS Python.
If you have any questions about the code or find any bugs
   or errors during use, please feel free to contact me.
If you have any questions about the original paper,
   please contact the authors of related paper.
"""
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import r2_score
from numpy import random
import time
import os
from sklearn import metrics
import matplotlib.pyplot as plt


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
        r2_adjusted = 1 - ((1 - metrics.r2_score(true, prediction)) * (n - 1)) / (n - p - 1)
    except:
        print("mse: {}, rmse: {}, mae: {}, mape: {}, r2: {}, rmsle: {}".format(mse, rmse, mae, mape, r2, rmsle))
        print('if you wanna get the value of r2_adjusted, you can define the number of features, '
              'which is the third parameter.')
        return mse, rmse, mae, mad, mape, r2, rmsle

    print("mse: {}, rmse: {}, mae: {}, mape: {}, r2: {}, r2_adjusted: {}, rmsle: {}".format(mse, rmse, mae, mape,
                                                                                            r2, r2_adjusted, rmsle))
    return mse, rmse, mae, mad, mape, r2, r2_adjusted, rmsle


def save_result(C, NumFea, NumWin, NumEnhan, mse, rmse, mae, mad, mape, r2, r2_adjusted, rmsle, t, save_result):
    # 计算行数，匹配 prediciton 的保存
    # print(precision)
    try:
        count = len(open(save_result, 'rU').readlines())
    except:
        count = 1

    if not os.path.exists(save_result):
        content = 'Count' + ',' + 'C' + ',' + 'NumFea' + ',' + 'NumWin' + ',' + 'NumEnhan' + ',' \
                  + 'mse' + ',' + 'rmse' + ',' + 'mae' + ',' \
                  + 'mad' + ',' + 'mape' + ',' \
                  + 'r2' + ',' + 'r2_adjusted' + ',' + 'rmsle' + ',' + 'Time'
        with open(save_result, 'a') as f:
            f.write(content)
            f.write('\n')
    content = str(count) + ',' + str(C) + ',' + str(NumFea) + ',' + str(NumWin) + "," + str(NumEnhan) + ',' \
              + str(mse) + ',' + str(rmse) + ',' + str(mae) + ',' \
              + str(mad) + ',' + str(mape) \
              + ',' + str(r2) + ',' + str(r2_adjusted) + ',' + str(rmsle) + ',' + str(t)
    with open(save_result, 'a') as f:
        f.write(content)
        f.write('\n')

    return count


def save_figures(ture, prediction, save_path, count):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                      '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                      '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                      '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

    plt.figure(num=None, figsize=(8, 2), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(range(len(prediction)), prediction, '-o', label='y_pred', color=color_sequence[0], linewidth=2)
    plt.plot(range(len(ture)), ture, '--o', label='y_true', color=color_sequence[2], linewidth=2)
    plt.grid()
    plt.legend()
    save_file = os.path.join(save_path, str(count) + '.png')
    plt.savefig(save_file, dpi=1000, bbox_inches='tight')
    plt.close()


def tansig(x):
    return (2 / (1 + np.exp(-2 * x))) - 1


def pinv(A, reg):
    return np.mat(reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)


'''
参数压缩
'''


def shrinkage(a, b):
    z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
    return z


'''
参数稀疏化
'''


def sparse_bls(A, b):
    lam = 0.001
    itrs = 50
    AA = np.dot(A.T, A)
    m = A.shape[1]
    n = b.shape[1]
    wk = np.zeros([m, n], dtype='double')
    ok = np.zeros([m, n], dtype='double')
    uk = np.zeros([m, n], dtype='double')
    L1 = np.mat(AA + np.eye(m)).I
    L2 = np.dot(np.dot(L1, A.T), b)
    for i in range(itrs):
        tempc = ok - uk
        ck = L2 + np.dot(L1, tempc)
        ok = shrinkage(ck + uk, lam)
        uk += ck - ok
        wk = ok
    return wk


def bls_regression(train_x, train_y, test_x, test_y, s, C, NumFea, NumWin, NumEnhan, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    u = 0
    WF = list()
    for i in range(NumWin):
        random.seed(i + u)
        WeightFea = 2 * random.randn(train_x.shape[1] + 1, NumFea) - 1;
        WF.append(WeightFea)
    #    random.seed(100)
    WeightEnhan = 2 * random.randn(NumWin * NumFea + 1, NumEnhan) - 1;
    time_start = time.time()
    H1 = np.hstack([train_x, 0.1 * np.ones([train_x.shape[0], 1])]);
    y = np.zeros([train_x.shape[0], NumWin * NumFea])
    WFSparse = list()
    distOfMaxAndMin = np.zeros(NumWin)
    meanOfEachWindow = np.zeros(NumWin)
    for i in range(NumWin):
        WeightFea = WF[i]
        A1 = H1.dot(WeightFea)
        scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(A1)
        A1 = scaler1.transform(A1)
        WeightFeaSparse = sparse_bls(A1, H1).T
        WFSparse.append(WeightFeaSparse)

        T1 = H1.dot(WeightFeaSparse)
        meanOfEachWindow[i] = T1.mean()
        distOfMaxAndMin[i] = T1.max() - T1.min()
        T1 = (T1 - meanOfEachWindow[i]) / distOfMaxAndMin[i]
        y[:, NumFea * i:NumFea * (i + 1)] = T1

    H2 = np.hstack([y, 0.1 * np.ones([y.shape[0], 1])])
    T2 = H2.dot(WeightEnhan)
    T2 = tansig(T2)
    T3 = np.hstack([y, T2])
    WeightTop = pinv(T3, C).dot(train_y)

    Training_time = time.time() - time_start
    print('Training has been finished!');
    print('The Total Training Time is : ', round(Training_time, 6), ' seconds')
    NetoutTrain = T3.dot(WeightTop)

    RMSE = np.sqrt((NetoutTrain - train_y).T * (NetoutTrain - train_y) / train_y.shape[0])
    MAPE = sum(abs(NetoutTrain - train_y)) / train_y.mean() / train_y.shape[0]
    R2 = r2_score(NetoutTrain, train_y)
    train_ERR = RMSE
    train_MAPE = MAPE
    train_R2 = R2
    print('Training RMSE is : ', RMSE)
    print('Training MAPE is : ', MAPE)
    print('Training R2 is: {}'.format(train_R2))
    time_start = time.time()
    HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0], 1])])
    yy1 = np.zeros([test_x.shape[0], NumWin * NumFea])
    for i in range(NumWin):
        WeightFeaSparse = WFSparse[i]
        TT1 = HH1.dot(WeightFeaSparse)
        TT1 = (TT1 - meanOfEachWindow[i]) / distOfMaxAndMin[i]
        yy1[:, NumFea * i:NumFea * (i + 1)] = TT1

    HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0], 1])])
    TT2 = tansig(HH2.dot(WeightEnhan));
    TT3 = np.hstack([yy1, TT2])
    NetoutTest = TT3.dot(WeightTop)

    mse, rmse, mae, mad, mape, r2, r2_adjusted, rmsle = reg_calculate(test_y, NetoutTest, features=train_x.shape[1])

    Testing_time = time.time() - time_start

    print('Testing has been finished!')
    # print(test_precision)
    save_file = os.path.join(save_path, 'BLS_result.csv')
    count = save_result(C, NumFea, NumWin, NumEnhan, mse, rmse, mae, mad, mape, r2, r2_adjusted, rmsle,
                        t=Testing_time, save_result=save_file)

    # save figure
    # save_fig = os.path.join(save_path, 'figure')
    # if not os.path.exists(save_fig):
    #     os.makedirs(save_fig)
    # save_figures(test_x, NetoutTest, save_path=save_fig, count=count)
    p_s = str(count) + '.csv'

    pred_path = os.path.join(save_path, 'prediction')
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    prediction_save = os.path.join(pred_path, p_s)
    np.savetxt(prediction_save, NetoutTest, delimiter=',')
    print('Save successful!!')