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

def save_results(epoch, lr, layer_numbers, hidden_layers, mse, rmse, mae, r2, save_file):

    save_file = save_file
    if not os.path.exists(save_file):
        content = 'epoch' + ',' + 'lr' + ',' + 'hidden_layer_number' + ',' + 'hidden_neurons' + ',' + 'mse' + \
                  ',' + 'rmse' + ',' + 'mae' + ',' + 'r2'
        with open(save_file, 'a') as f:
            f.write(content)
            f.write('\n')
    content = str(epoch) + ',' + str(lr) + "," + str(layer_numbers) + ',' + str(hidden_layers) + ',' + str(mse) + ',' + \
              str(rmse) + ',' + str(mae) + ',' + str(r2)
    with open(save_file, 'a') as f:
        f.write(content)
        f.write('\n')