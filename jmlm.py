import tqdm
import numpy as np
import pandas as pd
from numpy.linalg import pinv
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from load_data import load_data
from sklearn.metrics import pairwise_distances_argmin_min


class JMLM():
    def __init__(self):
        self.points = []
        self.F_p = []
        self.Jacobians = []


    def get_centroid(self, X, y, num_points):
        kmeans = KMeans(n_clusters=num_points).fit(X)
        current_points = kmeans.cluster_centers_
        neigh = KNeighborsClassifier(n_neighbors=3 if X.shape[0] > 3 else 1).fit(X,y)
        current_F_p = neigh.predict(current_points)
        # neigh = NearestNeighbors(n_neighbors=5).fit(X,y)
        # all_idx = neigh.kneighbors(current_points)[1]

        # current_F_p = []
        # for idx in all_idx:

        #     F_p = y[idx[0]]
        #     for i in idx[1:]:
        #         F_p += y[i]
        #     F_p /= len(idx)
        #     current_F_p.append(F_p)
        # current_F_p = np.stack(current_F_p)

        return current_points, current_F_p


    def calculate_jacobian(self, X, y, current_points, current_F_p):
        current_Jacobians = []
        for ci, centroid in enumerate(current_points):
            D_p = X - centroid
            D_f = y - current_F_p[ci]

            # nxN @ Nxm => nxm
            J_p = pinv(D_p) @ D_f
            current_Jacobians.append(J_p)
        return current_Jacobians

        
    def deep_train(self, X, y, iterations, max_num_points):
        self.train(X, y, 1, 1)
        X_train, X_vaild, y_train, y_vaild = train_test_split(X, y, test_size=0.20)
        new_X_train, new_y_train = X_train, y_train
        for _ in tqdm.tqdm(range(iterations)):
            max_acc = 0
            max_points = []
            max_F_p = []
            max_Jacobians = []
    
            for c in range(1, 1 + (max_num_points if len(new_X_train) > max_num_points else len(new_X_train))):
                current_points, current_F_p = self.get_centroid(new_X_train, new_y_train, c)
                current_Jacobians = self.calculate_jacobian(X_train, y_train, current_points, current_F_p)
                
                y_pred = self.test_predict(X_vaild, current_points, current_F_p, current_Jacobians)
                acc = accuracy_score(y_vaild.argmax(axis=1), y_pred)
                if acc < max_acc: continue
                max_acc = acc
                max_points = current_points
                max_F_p = current_F_p
                max_Jacobians = current_Jacobians
            
            try:
                self.update_points(np.append(self.points, max_points, axis=0), np.append(self.F_p, max_F_p, axis=0), self.Jacobians + max_Jacobians)
            except Exception as e:
                print(self.points.shape)
                print(np.array(max_points).shape)
                raise e

            new_X_train, new_y_train = self.clusters_loss(X_train, y_train, max_num_points)


    def test_predict(self, X, current_points, current_F_p, current_Jacobians):
        original_points = self.points
        original_F_p = self.F_p
        original_Jacobians = self.Jacobians
        self.update_points(np.append(self.points, current_points, axis=0), np.append(self.F_p, current_F_p, axis=0), self.Jacobians + current_Jacobians)
        y_pred, _ = self.forward(X)
        self.update_points(original_points, original_F_p, original_Jacobians)
        return y_pred.argmax(axis=1)


    def clusters_loss(self, X, y, max_num_points):
        neigh = KNeighborsClassifier(n_neighbors=1).fit(self.points, np.arange(len(self.points)))
        nearest_cis = neigh.predict(X)
        y_forward = self.forward(X)

        max_mse = 0
        max_X_train = []
        max_y_train = []

        for ci in range(len(self.points)):
            x_ci = X[nearest_cis==ci]
            if len(x_ci) < max_num_points: continue
            y_ci = y[nearest_cis==ci]
            y_ci_forward = y_forward[nearest_cis==ci]

            mse = mean_squared_error(np.array(y_ci), np.array(y_ci_forward))
            if mse < max_mse : continue
            max_mse = mse
            max_X_train = np.array(x_ci)
            max_y_train = np.array(y_ci)
        return max_X_train, max_y_train


    def jjlm_train(self, X, y, max_num_points):
        self.update_points(np.empty([0, X.shape[1]]), np.empty([0, y.shape[1]]), [])
        X_train, X_vaild, y_train, y_vaild = train_test_split(X, y, test_size=0.20)
        train_labels = []
        for label in y_train:
            train_labels.append(np.where(label==1)[0][0])
        
       
        for ci in np.unique(np.array(train_labels)):
            x_ci = X_train[train_labels==ci]
            y_ci = y_train[train_labels==ci]
            max_acc = 0
            max_points = []
            max_F_p = []
            max_Jacobians = []

            for c in tqdm.tqdm(range(1, max_num_points+1)):
                points, F_p  = self.get_centroid(x_ci, y_ci, c)
                Jacobians = self.calculate_jacobian(x_ci, y_ci, points, F_p)
                y_pred = self.test_predict(X_vaild, points, F_p, Jacobians)
                acc = accuracy_score(y_vaild.argmax(axis=1), y_pred)
                if acc < max_acc: continue
                max_points = points
                max_F_p = F_p
                max_Jacobians = Jacobians
                max_acc = acc

            
            self.update_points(np.append(self.points, max_points, axis=0), np.append(self.F_p, max_F_p, axis=0), self.Jacobians + max_Jacobians)
        print(self.points)

    def train(self, X, y, iterations, max_num_points, X_test, y_test):
        self.y_labels = np.unique(np.array(y), axis=0)
        self.y_labels = np.array(sorted(self.y_labels, key=lambda e: e.argmax()))
        X_train, X_vaild, y_train, y_vaild = train_test_split(X, y, test_size=0.20)
        # X_train, X_vaild, y_train, y_vaild = X, X_test, y, y_test
        max_acc = 0
        max_points = []
        max_F_p = []
        max_Jacobians = []

        # self.update_points(np.empty([0, X.shape[1]]), np.empty([0, y.shape[1]]), [])
        acc_list = []
        knn_acc_list = []
        for c in tqdm.tqdm(range(1, max_num_points+1)):
            for _ in range(iterations):
                points, F_p  = self.get_centroid(X_train, y_train, c)
                Jacobians = self.calculate_jacobian(X_train, y_train, points, F_p)
                self.update_points(points, F_p, Jacobians)
                y_pred, knn_pred = self.predict(X_vaild)
                acc = accuracy_score(y_vaild, y_pred)
                knn_acc = accuracy_score(y_vaild, knn_pred)
                # acc = accuracy_score(y_vaild.argmax(axis=1), y_pred)
                # knn_acc = accuracy_score(y_vaild.argmax(axis=1), knn_pred)
                acc_list.append(acc)
                knn_acc_list.append(knn_acc)

                if acc < max_acc: continue
                max_points = self.points
                max_F_p = self.F_p
                max_Jacobians = self.Jacobians
                max_acc = acc
        print('number of cent: ', len(max_points))
        print('Prediction Vaild in JMLM:', max_acc)
        self.update_points(max_points, max_F_p, max_Jacobians)
        return acc_list, knn_acc_list, acc_list


    def update_points(self, points, F_p, Jacobians):
        self.points = points
        self.F_p = F_p
        self.Jacobians = Jacobians


    def forward(self, X):
        neigh = KNeighborsClassifier(n_neighbors=1).fit(self.points, np.arange(len(self.points)))
        nearest_cis = neigh.predict(X)

        y_preds = []
        knn_preds = []
        for xi, ci in enumerate(nearest_cis):
            y_pred = self.F_p[ci] + (X[xi]-self.points[ci]) @ self.Jacobians[ci] 
            y_preds.append(y_pred) 
            knn_preds.append(self.F_p[ci]) 
        # 1xm + Nxm    
        return np.stack(y_preds), np.stack(knn_preds) 


    def predict(self, X):
        y_forward, knn_preds = self.forward(X)
        y_pred, _ = pairwise_distances_argmin_min(y_forward.reshape(-1, 1), self.y_labels.reshape(-1, 1))
        return y_pred, knn_preds
        # return y_forward.argmax(axis=1), knn_preds.argmax(axis=1)
def main():
    dataset = "iris"
    X_train, X_test, y_train, y_test = load_data(dataset, onehot=False)

    jmlm = JMLM()
    n_iterations = 10
    n_max_center = 8
    jmlm.train(X_train, y_train, n_iterations, n_max_center, X_test, y_test)
    # jmlm.jjlm_train(X_train, y_train, n_max_center)
    # deep_iteration = 100
    # max_kmeans_width = 10
    # jmlm.deep_train(X_train, y_train, deep_iteration, max_kmeans_width)
    

    y_pred, _ = jmlm.predict(X_train)
    acc = accuracy_score(y_train, y_pred)
    # y_pred=onehotencoder.transform(y_pred.reshape(-1, 1)).toarray()
    print('Train Accuracy :', acc)
    y_pred, _ = jmlm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # y_pred=onehotencoder.transform(y_pred.reshape(-1, 1)).toarray()
    print('Test Accuracy:', acc)

if __name__ == '__main__':
    for _ in range(5):
        main()
    