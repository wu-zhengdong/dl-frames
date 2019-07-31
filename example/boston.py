from sklearn import datasets

'''
The regression example of the Boston dataset.
'''

boston = datasets.load_boston()

X, y = boston.data, boston.target.reshape(-1, 1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

# 调用 frames 框架搭建好的 ann model.
from frames import DeeplearningRegression

# hidden layers 的神经元个数设置
hidden_layers = [128, 64, 32]

ann = DeeplearningRegression.ANN(hidden_layers, learning_rate=0.0001, epoch=10000)
# 训练模型
ann.fit(X_train, y_train)
# 计算模型分数
ann.score(X_test, y_test)
# 保存模型的结果图
import os
save_path = './example/save_png'
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_png = os.path.join(save_path, 'results.png')
ann.result_plot(X_test, y_test, save_file=save_png)
# 保存模型结果, 这里有两个参数，统计 dataset 是否使用了 标准化 和 pca 的预处理
results_file = './example/ann.csv'
ann.save_result(results_file, is_standard=False, is_PCA=False)