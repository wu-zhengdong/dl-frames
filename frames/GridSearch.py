import numpy as np
from .ELM import ELMRegression
from .ELM import ELMClassification
from sklearn import metrics
from . import tools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings('ignore')


def ElmRegressionGridSearch(X_train, X_test,
                           y_train, y_test, hidden_nodes=150, gap=10, is_show=False):
    ''' in order to search the best hidden number '''
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
                best_prediction = elm.predict(X_test)
            print("the neuron numbers of hidden layer: ", h)
            tools.reg_calculate(y_test, prediction)
            print('================================')

    print('============ The end!!! ==============')
    print("the neuron numbers of hidden layer: ", best_num)
    tools.reg_calculate(y_test, best_prediction)

    if is_show:
        plt.plot(range(len(best_prediction)), best_prediction, 'r--', label='prediction')
        plt.plot(range(len(y_test)), y_test, 'b--', label='true')
        plt.legend()
        plt.show()
    return best_prediction

def ElmClassificationGridSearch(X_train, X_test,
                           y_train, y_test, hidden_nodes=150, gap=10, is_show=False, delete_zero=False):
    ''' in order to search the best hidden number '''
    best_socre = -1000
    best_num = 0
    output_size = OneHotEncoder(sparse=False).fit_transform(y_train.reshape(-1, 1)).shape[1]
    for h in range(1, hidden_nodes):
        if h % gap == 0:
            elm = ELMClassification(hidden_nodes=h)
            elm.fit(X_train, y_train)
            prediction = elm.predict(X_test)
            score = metrics.r2_score(y_test, prediction)
            if best_socre < score:
                best_socre = score
                best_num = h
                best_prediction = elm.predict(X_test)
            print("the neuron numbers of hidden layer: ", h)
            tools.clf_calculate(y_test, prediction)
            print('================================')

    print('============ The end!!! ==============')
    print("the neuron numbers of hidden layer: ", best_num)
    tools.clf_calculate(y_test, best_prediction)

    if is_show:
        x, y = [], []
        elm = tools.confusion_matrix_result(y_test, best_prediction)
        if delete_zero:
            elm = elm[1:, 1:]
            for i in range(1, output_size):
                x.append('P' + str(i))
                y.append('T' + str(i))
        else:
            for i in range(output_size):
                x.append('P' + str(i))
                y.append('T' + str(i))

        pic = sns.heatmap(elm, annot=True, yticklabels=y, linewidths=0.5, xticklabels=x, cmap="YlGnBu")
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        sns.set(font_scale=1.5)
        # pic_save = pic.get_figure()
        # pic_save.savefig(save_file)
        plt.show()
        plt.close()
        print('Save the picture successfully!')
    return best_prediction