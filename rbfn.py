from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from kernel import Kernel, Gaussian
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, accuracy_score, mean_squared_error
import math
from load_data import load_data
import random
from sklearn.model_selection import train_test_split
from graphviz import Digraph

class RBFNetwork():
    def __init__(self, hidden_dim=2, kmeans_iter=20, threshold=0.1, n_max_center=100, lr=0.01, epoch=2000, kernel: Kernel = Gaussian()) -> None:
        self.hidden_dim = hidden_dim
        self.kmeans_iter = kmeans_iter
        self.threshold = threshold
        self.lr = lr
        self.epoch = epoch
        self.kernel = kernel
        self.n_max_center = n_max_center
        self.graph_data = []
        self.dot = Digraph(comment='Tree')

    def view(self):
        for node in self.graph_data:
            self.dot.node(node["index"], node["detail"])
            if node["parent"]:
                self.dot.edge(node["parent"], node["index"])
        self.dot.view()
        

    def set_center(self, X, y):
        kmeans = KMeans(n_clusters=self.hidden_dim, init='k-means++').fit(X)
        new_centers = kmeans.cluster_centers_
        
        clusters, _ = pairwise_distances_argmin_min(X, new_centers)
        new_sigma = []
        for idx, center in enumerate(new_centers):
            sigma = self.compute_sigma(X[clusters==idx], center)
            new_sigma.append(sigma)

        labels, _ = pairwise_distances_argmin_min(new_centers, X)
        new_labels = list(y[labels])
        return new_centers, new_labels, new_sigma

    def compute_sigma(self, X, center):
        sigma = (1 / len(X) * (np.sum(np.apply_along_axis(np.linalg.norm, 1, (X - center)))))
        if sigma == 0:
            return 0.1
        return sigma

    def compute_mse(self, X, y, X_c, y_c, new_centers, new_labels, new_sigma):
        y_forward, _ = self.forward(X, t_centers=new_centers, t_labels=new_labels, t_sigma=new_sigma)
        mse = mean_squared_error(np.array(y), np.array(y_forward))
        mse_c = []
        clusters, _ = pairwise_distances_argmin_min(X_c, new_centers)
        for ci in range(len(new_centers)):
            X_ci = X_c[clusters==ci]
            y_ci = y_c[clusters==ci]
            if len(X_ci) <= 0: continue
            y_preds = []
            for x in X_ci:
                y_pred = []
                for idx, center in enumerate(new_centers):
                    y_pred.append(self.kernel.compute(x, center, new_sigma[idx]))
                max_rbf_value_idx = np.array(y_pred).argmax(axis=0)
                y_preds.append(new_labels[max_rbf_value_idx])

            mse_c.append(mean_squared_error(np.array(y_ci), np.array(y_preds)))
        return mse, mse_c

    def initialize_parameters(self, X, y):
        self.output_dim = np.unique(np.array(y), axis=0).size
        self.centers = np.array([])
        self.sigma = []
        self.labels = []
        # self.output_dim = np.unique(np.array(y), axis=0).size
        # self.centers = np.random.random_sample(size=(1, X.shape[1]))
        # self.weights = list(np.random.random_sample(size=1))
        # self.sigma = [0.0001]

    def train(self, X, y):
        self.initialize_parameters(X, y)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
        train_data = [[X_train, y_train]]
        node_index = [0]
        layer = 0

        if len(self.graph_data) == 0:
            data_info = f"n_feature: {X.shape[1]}\nsamples: {X.shape[0]}"
            self.graph_data.append({"index" : str(0), "detail" : data_info, "parent" : None})

        while train_data and self.n_max_center > self.centers.shape[0]:
            print(f"=============={layer = }==============")
            next_data = []
            next_node_index = []
            for data, ni in zip(train_data, node_index):
                X_train, y_train = data[0], data[1]
                min_mse = float('inf')
                min_new_centers = None
                min_new_labels = None
                min_new_sigma = None
                min_mse_c = None
                print(f"{X_train.shape = }, {y_train.shape = }")
                for _ in range(self.epoch):
                    if X_train.shape[0] < self.hidden_dim: continue
                    new_centers, new_labels, new_sigma = self.set_center(X_train, y_train)
                    mse, mse_c = self.compute_mse(X, y, X_train, y_train, new_centers, new_labels, new_sigma)
                    print(f"{mse = }")
                    if mse > min_mse: continue
                    min_mse = mse
                    min_new_centers = new_centers
                    min_new_labels = new_labels
                    min_new_sigma = new_sigma
                    min_mse_c = mse_c
                
                if min_new_centers is None: continue
                # 同批資料但經過幾次iter找到較佳結果才更新到參數上
                try:
                    self.centers = np.concatenate((self.centers, min_new_centers)) if self.centers.size > 0 else min_new_centers
                except ValueError as e:
                    print(f"{self.centers.shape = }")
                    print(f"{min_new_centers.shape = }")
                    raise e
                self.labels += min_new_labels
                self.sigma += min_new_sigma
                if(len(self.centers) > self.n_max_center):break

                
                clusters, _ = pairwise_distances_argmin_min(X_train, min_new_centers)
                n_current = len(self.graph_data)
                for ci in range(len(min_new_centers)):
                    center_info = list(np.around(np.array(min_new_centers[ci]), 2))
                    center_mse = min_mse_c[ci]
                    n_sample = X_train[clusters==ci].shape[0]
                    node_detail = f"center: {center_info[0]}...{center_info[-1]}\nmse: {center_mse}\nsamples: {n_sample}"
                    self.graph_data.append({"index" : str(n_current), "detail" : node_detail, "parent" : str(ni)})
                    n_current += 1

                for ci, mse_c in enumerate(min_mse_c):
                    enough = mse_c < self.threshold
                    if not enough:
                        next_data.append([X_train[clusters==ci], y_train[clusters==ci]])
                        next_node_index.append(len(self.graph_data) - len(min_mse_c) + ci)
                if min_mse <= self.threshold: break
                print(f"current n centers: {len(self.centers)}")
            train_data = next_data
            node_index = next_node_index
            cur_mse, _ = self.compute_mse(X, y, X_train, y_train, self.centers, self.labels, self.sigma)
            if cur_mse <= self.threshold: break
            layer += 1

    def fit(self, X, y):
        self.train(X, y)
        return self

    def forward(self, X, t_centers=None, t_labels=None, t_sigma=None):
        
        centers = (np.concatenate((self.centers, t_centers)) if self.centers.size > 0 else t_centers) if t_centers is not None else self.centers
        labels = self.labels + t_labels if t_labels is not None else self.labels
        sigma = self.sigma + t_sigma if t_sigma is not None else self.sigma
        y_preds = []
        y_rbfs = []
 
        for x in X:
            y_pred = []
            y_rbf = []
            for idx, center in enumerate(centers):
                # y_pred.append(self.kernel.compute(x, center, self.sigma[idx]))
                rbf = self.kernel.compute(x, center, sigma[idx])
                # rbf = self.kernel.compute(x, center, self.sigma[idx]) if idx else 1
                # y_pred += self.weights[idx] * rbf
                y_pred.append(rbf)
                y_rbf.append(rbf)
            y_pred_max = np.array(y_pred).argmax()
            # interval = np.linspace(0, 1, num=self.output_dim, endpoint=False)
            # y_pred = interval[interval <= y_pred].argmax() if y_pred > 0 else 0
            # y_preds.append(y_pred)
            y_preds.append(labels[y_pred_max])
            y_rbfs.append(y_rbf)
        return y_preds, y_rbfs

    def predict(self, X):
        y_forward, _ = self.forward(X)
        return y_forward
        

def main():
    # X, y = datasets.make_moons(200, noise=0.3, random_state=42)
    
    X = np.array([[1, 1], [2, 1], [1, 2], [2, 2], [4, 4], [5, 4], [4, 5], [5, 5]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    X_test = np.array([[1.5, 1], [1.5, 2], [4.5, 4], [4.5, 5], [1.5, 1.5], [1, 1], [2, 1], [1, 2], [2, 2], [4, 4], [5, 4], [4, 5], [5, 5]])
    y_test = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1])

    # X, X_test, y, y_test = load_data('mini_mnist', onehot=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    rbfn = RBFNetwork(hidden_dim=2, n_max_center=10, threshold=0.1, epoch=10, lr=0.01)
    rbfn = rbfn.fit(X, y)
    
    rbfn.view()
    # print(f"{y_preds = }")
    print(f"{len(rbfn.centers) = }")

    y_train_preds = rbfn.predict(X)
    train_acc = accuracy_score(y, y_train_preds)
    print(f"{train_acc = }")
    y_test_preds = rbfn.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_preds)
    print(f"{test_acc = }")

    plt.xlabel("X0", fontsize=20)
    plt.ylabel("X1", fontsize=20) 
    plt.scatter(X[:,0], X[:,1], s=60, c=y)
    plt.scatter(rbfn.centers[:,0], rbfn.centers[:,1], s=60, c='r')
    for i, r in enumerate(rbfn.sigma):
        draw_circle = plt.Circle(list(rbfn.centers)[i], r, fill=False)
        ax.add_artist(draw_circle)
    plt.show()
    plt.savefig('D:/Applications/vscode/workspace/JMLM/outputs/rbfn.png')


if __name__ == "__main__":
    main()