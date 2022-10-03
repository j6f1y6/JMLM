import tqdm
import numpy as np
import pandas as pd
from numpy.linalg import pinv
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from load_data import load_data


class JMLM():
    def __init__(self):
        self.centroids = []
        self.F_p = []
        self.Jacobians = []


    def get_centroid(self, X, y, num_centroids):
        kmeans = KMeans(n_clusters=num_centroids).fit(X)
        current_centroids = kmeans.cluster_centers_
        neigh = KNeighborsClassifier(n_neighbors=3 if X.shape[0] > 3 else 1).fit(X,y)
        current_F_p = neigh.predict(current_centroids)
        return current_centroids, current_F_p


    def calculate_jacobian(self, X, y, current_centroids, current_F_p):
        current_Jacobians = []
        for ci, centroid in enumerate(current_centroids):
            D_p = X - centroid
            D_f = y - current_F_p[ci]

            # nxN @ Nxm => nxm
            J_p = pinv(D_p) @ D_f
            current_Jacobians.append(J_p)
        return current_Jacobians

        
    def deep_train(self, X, y, iterations, max_num_centroids):
        self.train(X, y, 5, 5)
        X_train, X_vaild, y_train, y_vaild = train_test_split(X, y, test_size=0.20)
        new_X_train, new_y_train = X_train, y_train
        for _ in tqdm.tqdm(range(iterations)):
            max_acc = 0
            max_centroids = []
            max_F_p = []
            max_Jacobians = []
    
            for c in range(1, 1 + (max_num_centroids if len(new_X_train) > max_num_centroids else len(new_X_train))):
                current_centroids, current_F_p = self.get_centroid(new_X_train, new_y_train, c)
                current_Jacobians = self.calculate_jacobian(X_train, y_train, current_centroids, current_F_p)
                
                y_pred = self.deep_predict(X_vaild, current_centroids, current_F_p, current_Jacobians)
                acc = accuracy_score(y_vaild.argmax(axis=1), y_pred)
                if acc < max_acc: continue
                max_acc = acc
                max_centroids = current_centroids
                max_F_p = current_F_p
                max_Jacobians = current_Jacobians
            
            try:
                self.update_centroids(np.append(self.centroids, max_centroids, axis=0), np.append(self.F_p, max_F_p, axis=0), self.Jacobians + max_Jacobians)
            except Exception as e:
                print(self.centroids.shape)
                print(np.array(max_centroids).shape)
                raise e

            new_X_train, new_y_train = self.clusters_loss(X_train, y_train, max_num_centroids)


    def deep_predict(self, X, current_centroids, current_F_p, current_Jacobians):
        original_centroids = self.centroids
        original_F_p = self.F_p
        original_Jacobians = self.Jacobians
        self.update_centroids(np.append(self.centroids, current_centroids, axis=0), np.append(self.F_p, current_F_p, axis=0), self.Jacobians + current_Jacobians)
        y_pred = self.forward(X)
        self.update_centroids(original_centroids, original_F_p, original_Jacobians)
        return y_pred.argmax(axis=1)


    def clusters_loss(self, X, y, max_num_centroids):
        neigh = KNeighborsClassifier(n_neighbors=1).fit(self.centroids, np.arange(len(self.centroids)))
        nearest_cis = neigh.predict(X)
        y_forward = self.forward(X)

        max_mse = 0
        max_X_train = []
        max_y_train = []

        for ci in range(len(self.centroids)):
            x_ci = X[nearest_cis==ci]
            if len(x_ci) < max_num_centroids: continue
            y_ci = y[nearest_cis==ci]
            y_ci_forward = y_forward[nearest_cis==ci]

            mse = mean_squared_error(np.array(y_ci), np.array(y_ci_forward))
            if mse < max_mse : continue
            max_mse = mse
            max_X_train = np.array(x_ci)
            max_y_train = np.array(y_ci)
        return max_X_train, max_y_train


    def jjlm_train(self):
        pass


    def train(self, X, y, iterations, max_num_centroids):
        X_train, X_vaild, y_train, y_vaild = train_test_split(X, y, test_size=0.20)
        max_acc = 0
        max_centroids = []
        max_F_p = []
        max_Jacobians = []

        self.update_centroids(np.empty([0, X.shape[1]]), np.empty([0, y.shape[1]]), [])

        for c in tqdm.tqdm(range(1, max_num_centroids+1)):
            for _ in range(iterations):
                centroids, F_p  = self.get_centroid(X_train, y_train, c)
                Jacobians = self.calculate_jacobian(X_train, y_train, centroids, F_p)
                self.update_centroids(centroids, F_p, Jacobians)
                y_pred = self.predict(X_vaild)
                acc = accuracy_score(y_vaild.argmax(axis=1), y_pred)
                if acc < max_acc: continue
                max_centroids = self.centroids
                max_F_p = self.F_p
                max_Jacobians = self.Jacobians
                max_acc = acc
        print('number of cent: ', len(max_centroids))
        print('Prediction Vaild in JMLM:', max_acc)
        self.update_centroids(max_centroids, max_F_p, max_Jacobians)


    def update_centroids(self, centroids, F_p, Jacobians):
        self.centroids = centroids
        self.F_p = F_p
        self.Jacobians = Jacobians


    def forward(self, X):
        neigh = KNeighborsClassifier(n_neighbors=1).fit(self.centroids, np.arange(len(self.centroids)))
        nearest_cis = neigh.predict(X)

        y_preds = []
        for xi, ci in enumerate(nearest_cis):
            y_pred = self.F_p[ci] + (X[xi]-self.centroids[ci]) @ self.Jacobians[ci] 
            y_preds.append(y_pred) 
        # 1xm + Nxm    
        return np.stack(y_preds)


    def predict(self, X):
        return self.forward(X).argmax(axis=1)


if __name__ == '__main__':
    dataset = "mnist"
    X_train, X_test, y_train, y_test = load_data(dataset)

    jmlm = JMLM()
    # n_iterations = 10
    # n_max_center = 100
    # jmlm.train(X_train, y_train, n_iterations, n_max_center)
    deep_iteration = 100
    max_kmeans_width = 10
    jmlm.deep_train(X_train, y_train, deep_iteration, max_kmeans_width)
    
    y_pred = jmlm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # y_pred=onehotencoder.transform(y_pred.reshape(-1, 1)).toarray()
    print('Prediction Accuracy after JMLM:', acc)