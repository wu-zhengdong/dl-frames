from frames.CV import AnnCV
from frames.DeeplearningClassification2 import ANN

parameters = {
    'hidden_layers' : [[32, 32,32,32,32,32], [64], [32,64]],
    'learning_rate' : [1e-3, 1e-5],
    'dropout' : [0.2, 0.5, 0.7],
    'activate_function' : ['relu', 'sigmoid'],
    'weight_decay' : [1e-5, 1e-7],
    'epoch' : [200, 2000],
    'batch_size' : [128, 256],
    'is_standard' : False,
    'Dimensionality_reduction_method' : 'PCA',
    'save_path' : './Classification_results/ANN',
    'device' : 1,
    'use_more_gpu' : False
}

from sklearn import datasets

boston = datasets.load_digits()
X, y = boston.data, boston.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

cv = AnnCV(ANN, parameters, X_train, y_train, X_test, y_test)

cv.fit()

