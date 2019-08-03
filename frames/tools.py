import numpy as np
from sklearn import metrics
import os

def calculate(true, prediction):
    mse = metrics.mean_squared_error(true, prediction)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(true, prediction)
    r2 = metrics.r2_score(true, prediction)

    print("mse: {}, rmse: {}, mae: {}, r2: {}".format(mse, rmse, mae, r2))
    return mse, rmse, mae, r2


def save_ann_results(epoch, batch_size, lr, dropout, layer_numbers, hidden_layers, activate_function, mse, rmse, mae, r2, is_standrad, is_PCA, save_file):

    save_file = save_file
    if not os.path.exists(save_file):
        content = 'epoch' + ',' + 'batch_size' + ',' + 'lr' + ',' + 'dropout' + ',' + 'hidden_layer_number' + ',' + 'hidden_neurons' + ','\
                  + 'activate function' + ',' + 'mse' + ',' + 'rmse' + ',' + 'mae' + ',' + 'r2' + ',' + 'is_standard' + ',' + 'is_PCA'
        with open(save_file, 'a') as f:
            f.write(content)
            f.write('\n')
    content = str(epoch) + ',' + str(batch_size) + ',' + str(lr) + "," + str(dropout) + ',' + str(layer_numbers) + ',' \
              + str(hidden_layers) + ',' + str(activate_function) + ',' + str(mse) + ',' + str(rmse) + ',' + str(mae) + ',' + str(r2) + ',' + str(is_standrad) + ',' + str(is_PCA)
    with open(save_file, 'a') as f:
        f.write(content)
        f.write('\n')


def save_cnn_results(epoch, batch_size, lr, dropout, conv_layers, channle_numbers, conv_kernel_size, conv_stride, pooling_size, pooling_stride,
                 flatten, activate_function, mse, rmse, mae, r2, is_standrad, is_PCA, save_file):

    save_file = save_file
    if not os.path.exists(save_file):
        content = 'epoch' + ',' + 'batch_size' + ',' + 'lr' + ',' + 'dropout' + ',' + 'conv_layers' + ',' + \
                  'channle_numbers' + ',' + 'conv_kernel_size' + ',' + 'conv_stride' + ',' + 'pooling_kernel_size' + \
                  ',' + 'pooling_stride' + ',' + 'flatten' + ',' + 'activate function' + ',' + 'mse' + ',' + 'rmse' + \
                  ',' + 'mae' + ',' + 'r2' + ',' + 'is_standard' + ',' + 'is_PCA'
        with open(save_file, 'a') as f:
            f.write(content)
            f.write('\n')
    content = str(epoch) + ',' + str(batch_size) + ',' + str(lr) + "," + str(dropout) + ',' + str(conv_layers) + ',' + str(channle_numbers) + ',' \
              + str(conv_kernel_size) + ',' + str(conv_stride) + ',' + str(pooling_size) + ',' + str(pooling_stride) + ',' \
              + str(flatten) + ',' + str(activate_function) + ',' + str(mse) + ',' + str(rmse) + ',' + str(mae) + ',' + \
              str(r2) + ',' + str(is_standrad) + ',' + str(is_PCA)
    with open(save_file, 'a') as f:
        f.write(content)
        f.write('\n')