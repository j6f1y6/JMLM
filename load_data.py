import cv2
import math
import numpy as np
import pandas as pd
from keras.datasets import cifar10
from keras.datasets import mnist
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler

def add_gaussian_noise(X_imgs):
    gaussian_noise_imgs = []
    for img in X_imgs:
        mean = 0
        var = 10
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, img.shape)

        noisy_image = np.zeros(img.shape, np.float32)

        if len(img.shape) == 2:
            noisy_image = img + gaussian
        else:
            noisy_image[:, :, 0] = img[:, :, 0] + gaussian
            noisy_image[:, :, 1] = img[:, :, 1] + gaussian
            noisy_image[:, :, 2] = img[:, :, 2] + gaussian

        cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
        noisy_image = noisy_image.astype(np.uint8)
        gaussian_noise_imgs.append(noisy_image)
    return np.array(gaussian_noise_imgs, dtype = np.float32)


def load_data(dataset, onehot=True, normalization=True, pca=1, lda=False, noise=False, asmpt_target=0, classification=True):
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
    
    elif dataset == "ICU":
        test_data = pd.read_csv("D:/Applications/vscode/workspace/JMLM/datasets/ICU/ICU_ALL_DATA_test.txt", sep="	", header=None) 
        train_data = pd.read_csv("D:/Applications/vscode/workspace/JMLM/datasets/ICU/ICU_ALL_DATA_train.txt", sep="	", header=None) 
        # data = train_data.append(test_data, ignore_index=True)
        # data = [train_data, test_data]
        # data = pd.concat(data)
        # X = data.iloc[:, 1:10].to_numpy()
        # y = data.iloc[:, 0].to_numpy()
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        X_train, X_test, y_train, y_test  = train_data.iloc[:, 1:10].to_numpy(), test_data.iloc[:, 1:10].to_numpy(), train_data.iloc[:, 0].to_numpy(), test_data.iloc[:, 0].to_numpy()

    elif dataset == "bridge":
        raw = pd.read_excel("D:/Applications/vscode/workspace/JMLM/datasets/bridge_reinforcement/2021-Taoyuan-??????????????????-?????????.xlsx")
        data = raw[0:6140]
        # X = data[['????????????', '????????????(?????????+1)', '????????????', 'D', 'E', 'R', 'U']]
        # X = data[['????????????', '????????????(?????????+1)', 'U', '??????', '????????????(m)', '????????????(m)', '??????(???)', '??????']]
        X = data[['????????????', '????????????(?????????+1)', 'U', '??????', '????????????(m)', '????????????(m)', '??????(???)', '????????????', '??????']]
        # X = data[['??????', '????????????', '????????????(?????????+1)', '????????????', 'D', 'E', 'R', 'U']]
        y = data["????????????"].to_numpy()
        X = X.dropna().copy()
        print(X.shape)
        labelencoder = LabelEncoder()
        X['U'] = X['U'].astype("float")
        X['??????(???)'] = X['??????(???)'].astype("string")
        # X['??????'] = labelencoder.fit_transform(X['??????'])
        X['????????????'] = labelencoder.fit_transform(X['????????????'])
        # X['????????????'] = labelencoder.fit_transform(X['????????????'])
        X['??????'] = labelencoder.fit_transform(X['??????'])
        X['????????????'] = labelencoder.fit_transform(X['????????????'])
        
        X['??????(???)'] = labelencoder.fit_transform(X['??????(???)'])
        
        X = X.to_numpy()
        print(X.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    elif dataset == "mini_mnist":
        X, y = load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    elif dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        if noise:
            X_train = add_gaussian_noise(X_train)

    elif dataset == "cifar10":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        
        # y_train = np.ravel(y_train)
        # y_test = np.ravel(y_test)

    elif dataset == "asmpt_train":
        data = pd.read_csv("D:/Applications/vscode/workspace/JMLM/datasets/asmpt/Train.csv")
        Train_data = data.loc[data['TrainTest'] == 'Train']
        Test_data = data.loc[data['TrainTest'] == 'Test']
        X_train = Train_data.loc[:,Train_data.columns.str.startswith('feature_')].to_numpy()
        y_train = Train_data['target_' + str(asmpt_target)].to_numpy()
        X_test = Test_data.loc[:,Test_data.columns.str.startswith('feature_')].to_numpy()
        y_test = Test_data['target_' + str(asmpt_target)].to_numpy()

    X_train = X_train.reshape(-1, math.prod(X_train.shape[1:]))
    X_test = X_test.reshape(-1, math.prod(X_train.shape[1:]))
    if normalization:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


    if classification:
        labelencoder = LabelEncoder()
        y_train = labelencoder.fit_transform(y_train)
        y_test = labelencoder.transform(y_test)

    if onehot:
        onehotencoder = OneHotEncoder()
        y_train = onehotencoder.fit_transform(y_train.reshape(-1, 1)).toarray()
        y_test = onehotencoder.transform(y_test.reshape(-1, 1)).toarray()

    if pca < 1 and pca > 0:
        pca = PCA(n_components=int(X_train.shape[1]*pca))
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    if lda:
        clf = LinearDiscriminantAnalysis()
        X_train = clf.fit_transform(X_train, y_train.argmax(axis=1) if onehot else y_train)
        X_test = clf.transform(X_test)


    return  X_train, X_test, y_train, y_test