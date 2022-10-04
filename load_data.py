import math
import numpy as np
import pandas as pd
from keras.datasets import cifar10
from keras.datasets import mnist
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_data(dataset):
    if dataset == "iris":
        data = pd.read_csv("D:/Applications/vscode/workspace/JMLM/datasets/Iris/Iris.csv")
        X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].to_numpy()
        y = data["Species"].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    elif dataset == "mini_mnist":
        X, y = load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    elif dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    elif dataset == "cifar10":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        
        # y_train = np.ravel(y_train)
        # y_test = np.ravel(y_test)

    X_train = X_train.reshape(-1, math.prod(X_train.shape[1:]))
    X_test = X_test.reshape(-1, math.prod(X_train.shape[1:]))

    labelencoder = LabelEncoder()
    y_train_label = labelencoder.fit_transform(y_train)
    y_test = labelencoder.transform(y_test)

    onehotencoder = OneHotEncoder()
    y_train = onehotencoder.fit_transform(y_train_label.reshape(-1, 1)).toarray()

    return  X_train, X_test, y_train, y_test



if __name__ == "__main__":
    load_data("iris")