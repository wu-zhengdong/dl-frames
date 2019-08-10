import numpy as np
from .ELM import ELMRegression
from sklearn import metrics
from . import tools
import matplotlib.pyplot as plt


def search_best_hidden_num(X_train, X_test,
                           y_train, y_test, hidden_nodes=150, gap=10, is_show=False):
    best_socre = -1000
    best_num = 0
    for h in range(1, hidden_nodes):
        if h % gap == 0:
            elm = ELMRegression(h)
            elm.fit(X_train, y_train)
            prediction = elm.predict(X_test)
            score = metrics.r2_score(y_test, prediction)
            if best_socre < score:
                best_socre = score
                best_num = h
                prediction = elm.predict(X_test)
            print("the neuron numbers of hidden layer: ", h)
            tools.reg_calculate(y_test, prediction)
            print('================================')

    print('============ The end!!! ==============')
    print("MSE: ", metrics.mean_squared_error(y_test, prediction))
    print("MAE: ", metrics.mean_absolute_error(y_test, prediction))
    print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, prediction)))
    print("R2:", best_socre)
    print("the neuron numbers of hidden layer: ", best_num)

    if is_show:
        plt.plot(range(len(prediction)), prediction, 'r--', label='prediction')
        plt.plot(range(len(y_test)), y_test, 'b--', label='true')
        plt.legend()
        plt.show()
    return prediction