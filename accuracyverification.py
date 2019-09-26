from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris = load_iris()

def main():
    x_train, x_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)


    print(knn.score(x_test, y_test))

    return

if __name__ == "__main__":main()
