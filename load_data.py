import math
import numpy as np
import pandas as pd
from keras.datasets import cifar10
from keras.datasets import mnist
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_data(dataset, onehot=True):
    if dataset == "iris_test":
        data = pd.read_csv("D:/Applications/vscode/workspace/JMLM/datasets/Iris/Iris.csv")
        X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].to_numpy()
        y = data["Species"].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        labelencoder = LabelEncoder()
        y_train = labelencoder.fit_transform(y_train)
        y_test = labelencoder.transform(y_test)
        return X_train, X_test, y_train, y_test

    if dataset == "iris":
        data = pd.read_csv("D:/Applications/vscode/workspace/JMLM/datasets/Iris/Iris.csv")
        X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].to_numpy()
        y = data["Species"].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    
    elif dataset == "wisconsin":
        data = pd.read_csv("D:/Applications/vscode/workspace/JMLM/datasets/wisconsin/data.csv")
        X = data[["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]].to_numpy()
        y = data["diagnosis"].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    elif dataset == "bridge":
        raw = pd.read_excel("D:/Applications/vscode/workspace/JMLM/datasets/bridge_reinforcement/2021-Taoyuan-橋梁補強資料-林聖國.xlsx")
        data = raw[0:6140]
        X = data[['結構型式', '總橋孔數(落墩數+1)', '構件名稱', 'D', 'E', 'R', 'U']]
        # X = data[['位置', '結構型式', '總橋孔數(落墩數+1)', '構件名稱', 'D', 'E', 'R', 'U']]
        y = data["維修工法"].to_numpy()
        X = X.dropna().copy()
        labelencoder = LabelEncoder()
        # X['位置'] = labelencoder.fit_transform(X['位置'])
        X['結構型式'] = labelencoder.fit_transform(X['結構型式'])
        X['構件名稱'] = labelencoder.fit_transform(X['構件名稱'])
        X = X.to_numpy()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

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
    y_train = labelencoder.fit_transform(y_train)
    y_test = labelencoder.transform(y_test)
    if onehot:
        onehotencoder = OneHotEncoder()
        y_train = onehotencoder.fit_transform(y_train.reshape(-1, 1)).toarray()
        y_test = onehotencoder.transform(y_test.reshape(-1, 1)).toarray()

    return  X_train, X_test, y_train, y_test