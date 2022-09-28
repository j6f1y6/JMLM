import tqdm
import numpy as np
import pandas as pd
from numpy.linalg import pinv
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from load_data import load_data


class JMLM():
    def __init__(self):
        self.centroids = []
        self.F_p = []
        self.Jacobians = []

    def set_centroid(self, X, y, num_centroids):
        kmeans = KMeans(n_clusters=num_centroids).fit(X)
        self.centroids = kmeans.cluster_centers_
        neigh = KNeighborsClassifier(n_neighbors=3).fit(X,y)
        self.F_p = neigh.predict(self.centroids)
        self.Jacobians = []
        for ci, centroid in enumerate(self.centroids):
            D_p = X - centroid
            D_f = y - self.F_p[ci]

            # nxN @ Nxm => nxm
            J_p = pinv(D_p) @ D_f
            self.Jacobians.append(J_p)

    def train(self, X, y, iterations, max_num_centroids):
        X_train, X_vaild, y_train, y_vaild = train_test_split(X, y, test_size=0.20)
        max_acc = 0
        max_centroids = []
        max_F_p = []
        max_Jacobians = []

        for c in tqdm.tqdm(range(1, max_num_centroids+1)):
            for _ in range(iterations):
                self.set_centroid(X_train, y_train, c)
                y_pred = self.predict(X_vaild)
                acc = accuracy_score(y_vaild.argmax(axis=1), y_pred)
                if acc < max_acc: continue
                max_centroids = self.centroids
                max_F_p = self.F_p
                max_Jacobians = self.Jacobians
                max_acc = acc
        print('number of cent: ', len(max_centroids))
        print('Prediction Vaild in JMLM:', max_acc)
        self.centroids = max_centroids
        self.F_p = max_F_p
        self.Jacobians = max_Jacobians

    def predict(self, X):
        neigh = KNeighborsClassifier(n_neighbors=1).fit(self.centroids, np.arange(len(self.centroids)))
        nearest_cis = neigh.predict(X)

        y_preds = []
        for xi, ci in enumerate(nearest_cis):
            y_pred = self.F_p[ci] + (X[xi]-self.centroids[ci]) @ self.Jacobians[ci] 
            y_preds.append(y_pred)
        # print(np.stack(y_preds))
        # 1xm + Nxm    
        return np.stack(y_preds).argmax(axis=1)


if __name__ == '__main__':
    dataset = "iris"
    X_train, X_test, y_train, y_test = load_data(dataset)

    n_iterations = 20
    n_max_center = 50

    jmlm = JMLM()
    jmlm.train(X_train, y_train, n_iterations, n_max_center)
    y_pred = jmlm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # y_pred=onehotencoder.transform(y_pred.reshape(-1, 1)).toarray()
    print('Prediction Accuracy after JMLM:', acc)