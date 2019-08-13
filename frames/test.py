from frames.GridSearch import ElmClassificationGridSearch
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

elm = ElmClassificationGridSearch(X_train, X_test, y_train, y_test, hidden_nodes=200, gap=10, is_show=True)