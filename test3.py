import numpy as np
from sklearn import datasets

digits = datasets.load_digits()

X, y = digits.data, digits.target

print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

from sklearn.preprocessing import StandardScaler

standard = StandardScaler()
standard.fit(X_train)
X_train_standard = standard.transform(X_train)
X_test_standard = standard.transform(X_test)


# from frames.DeeplearningClassification2 import CNN
#
# cnn = CNN(learning_rate=[1e-3, 1e-5, 1e-7], conv_stride=1, kernel_size=3, pooling_size=2, pool_stride=2,
#                  channel_numbers=[64, 32], flatten=1024, activate_function='relu', dropout=0, weight_decay=1e-8,
#                  momentum=0.5, device=0, epoch=2000, batch_size=128,
#                  use_more_gpu=False, is_standard=False,
#                  Dimensionality_reduction_method='None', save_path='./Classification_result/CNN_Results')
#
# cnn.fit(X_train_standard, y_train, X_test_standard, y_test)
#
# cnn.score()
#
# cnn.save()

# from frames.DeeplearningClassification2 import ANN
#
# ann = ANN(learning_rate=[1e-3, 1e-5, 1e-7], hidden_layers=[32, 64], weight_decay=1e-8,
#           save_path='./Classification_result/ANN_Results')
#
# ann.fit(X_train_standard, y_train, X_test_standard, y_test)
# ann.score()
# ann.save()


print(1e-3/1.1)

h = [32, 64, 128]
from itertools import product
hidden_layers = list(product(h, repeat=2)) # tuple
hl = []

for i in range(len(hidden_layers)):
    list_h = list(hidden_layers[i])
    hl.append(list_h)
print(len(hl))