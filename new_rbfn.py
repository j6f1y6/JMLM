from turtle import forward
import tqdm
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
import os
from datetime import datetime

class RBFNetwork():
    def __init__(self, hidden_dim=2, kmeans_iter=20, threshold=0.1, n_max_center=100, lr=0.5, epoch=2000, epoch_p=100, kernel: Kernel = Gaussian(), root=f'./outputs/{datetime.now().strftime("%Y%m%d%p%I%M%S")}/') -> None:
        self.hidden_dim = hidden_dim
        self.kmeans_iter = kmeans_iter
        self.threshold = threshold
        self.lr = lr
        self.epoch = epoch
        self.kernel = kernel
        self.n_max_center = n_max_center
        self.root = root
        self.epoch_p = epoch_p


    def add_center(self, X, y):
        kmeans = KMeans(n_clusters=self.hidden_dim, init='k-means++').fit(X)
        new_centers = kmeans.cluster_centers_
        
        clusters, _ = pairwise_distances_argmin_min(X, new_centers)
        new_sigma = []
        for idx, center in enumerate(new_centers):
            sigma = self.compute_sigma(X[clusters==idx], center)
            new_sigma.append(sigma)

        return new_centers, new_sigma


    def compute_sigma(self, X, center):
        sigma = (1 / len(X) * (np.sum(np.apply_along_axis(np.linalg.norm, 1, (X - center)))))
        if sigma == 0:
            return 0.1
        return sigma


    def get_interval_prediction(self, y_forward):
        interval = np.linspace(0, 1, num=self.output_dim, endpoint=False)
        # print(f"{interval = }")
        y_preds = []
        for idx, f_val in enumerate(y_forward):
            y_preds.append(interval[interval <= f_val].argmax())
        return y_preds


    def initialize_parameters(self, X, y):
        self.output_dim = np.unique(np.array(y), axis=0).size
        self.centers = np.array([])
        self.sigma = []


    def cluster_forward(self, ci, X):
        y_forward = []
        for x in X:
            forward = self.kernel.compute(x, self.centers[ci], self.sigma[ci])
            y_forward.append(forward)
        return np.array(y_forward)


    def cluster_backward(self, ci, X, y_forward, y):
        center = self.centers[ci]
        sigma = self.sigma[ci]
        # v_target = v_target / self.output_dim
        v_pred = self.get_interval_prediction(y_forward)
        output_grad = y - v_pred
        self.centers[ci] = center + self.lr / (sigma ** 2) * ((X - center).T @ (output_grad * y_forward)) 
        self.sigma[ci] = sigma + self.lr / (sigma ** 3) * np.linalg.norm(X - center, axis=1).reshape(1, -1) @ (output_grad * y_forward)


    def vote_predict(self, X):
        y_pred_list = []
        for ci in range(len(self.centers)):
            y_forward = self.cluster_forward(ci, X)
            y_pred = self.get_interval_prediction(y_forward)
            y_pred_list.append(y_pred)
        y_preds = np.array([np.bincount(ai).argmax() for ai in np.array(y_pred_list).T])
        return y_preds
        
        
    def train_one_epoch(self, X, y, clusters=None):
        if clusters is not None:
            for ci in np.unique(np.array(clusters), axis=0):
                y_forward = self.cluster_forward(ci, X[clusters==ci])
                self.cluster_backward(ci, X[clusters==ci], y_forward, y[clusters==ci])
            # for idx, x in enumerate(X):
            #     y_forward = self.kernel.compute(x, self.centers[clusters[idx]], self.sigma[clusters[idx]])
            #     self.backward(x.reshape(1, -1), y_forward, y[idx])


        # # 每個群心對全部
        # for ci in range(len(self.centers)):
        #     y_forward = self.cluster_forward(ci, X)
        #     self.cluster_backward(ci, X, y_forward, y)
        
        # # 每個群心對該群更新
        # clusters, _ = pairwise_distances_argmin_min(X, self.centers)
        # for ci in range(len(self.centers)):
        #     y_forward = self.cluster_forward(ci, X[clusters==ci])
        #     self.cluster_backward(ci, X[clusters==ci], y_forward, y[clusters==ci])
        
        # 每群有outputdim個std去更新
        # clusters, _ = pairwise_distances_argmin_min(X, self.centers)
        # for ci in range(len(self.centers)):
        #     y_forward = self.cluster_forward(ci, X[clusters==ci])
        #     for class_idx in range(self.output_dim):
        #         self.cluster_backward(ci, class_idx, X[clusters==ci], y_forward[class_idx], y[clusters==ci][class_idx])
       
        # # 一個接一個
        # all_index = np.arange(len(X))
        # np.random.shuffle(all_index)
        # for ri in all_index:
            
        #     xi = X[ri].reshape(1, -1)
        #     yi = y[ri]
        #     y_forward = self.forward(xi)
        #     self.backward(xi, y_forward[0], yi)


    def train(self, X, y):
        self.initialize_parameters(X, y)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
        train_data = [[X_train, y_train]]
        node_index = [0]
        layer = 0

        while train_data and self.n_max_center > self.centers.shape[0]:
            print(f"=============={layer = }==============")
            next_data = []
            for data, ni in zip(train_data, node_index):
                X_train, y_train = data[0], data[1]
                # print(f"{X_train.shape = }, {y_train.shape = }")
                
                if X_train.shape[0] < self.hidden_dim: continue
                new_centers, new_sigma = self.add_center(X_train, y_train)
            
                self.centers = np.concatenate((self.centers, new_centers)) if self.centers.size > 0 else new_centers
                self.sigma += new_sigma

                clusters, _ = pairwise_distances_argmin_min(X_train, self.centers)

                self.file_name = 0
                for i in tqdm.tqdm(range(self.epoch_p)):
                    # self.train_one_epoch(X_train, y_train)
                    self.train_one_epoch(X_train, y_train, clusters=clusters)
                    if i % 100 == 0:
                        self.draw_graph(X, y, str(i)) 

                if(len(self.centers) > self.n_max_center): break
    
                for ci in range(len(self.centers)):
                    xci = X_train[clusters==ci]
                    yci = y_train[clusters==ci]
                    if xci.size == 0: continue
                    ycif = self.predict(xci)
                    yciacc = accuracy_score(yci, ycif)
                    print(f"{self.centers[ci]}, {np.unique(np.array(yci), axis=0) = }, {yciacc = }")
                print("======================")

                print(f"current n centers: {len(self.centers)}")
            train_data = next_data
            layer += 1

    def fit(self, X, y):
        self.train(X, y)
        return self

    def forward(self, X, t_centers=None, t_sigma=None):
        centers = t_centers if t_centers is not None else self.centers
        sigma = t_sigma if t_sigma is not None else self.sigma
        y_preds = []

        clusters, _ = pairwise_distances_argmin_min(X, centers)
        # print(clusters)
 
        for idx, x in enumerate(X):
            rbf = self.kernel.compute(x, centers[clusters[idx]], sigma[clusters[idx]])
            y_preds.append(rbf)
        return y_preds

    def backward(self, x, y_forward, y):
        
        clusters, _ = pairwise_distances_argmin_min(x, self.centers)
        center = self.centers[clusters[0]]
        sigma = self.sigma[clusters[0]]
        # y = y / self.output_dim
        y_pred = self.get_interval_prediction([y_forward])
        self.centers[clusters[0]] = center + self.lr * (y - y_pred) * y_forward * (x - center) / (sigma ** 2) 
        self.sigma[clusters[0]] = sigma + self.lr * (y - y_pred) * y_forward * np.linalg.norm(x - center) / (sigma ** 3)
        # self.centers[clusters[0]] = center + self.lr * (y - y_forward) * y_forward * (x - center) / (sigma ** 2) 
        # self.sigma[clusters[0]] = sigma + self.lr * (y - y_forward) * y_forward * np.linalg.norm(x - center) / (sigma ** 3)
        
    def predict(self, X):
        y_forward = self.forward(X)
        y_pred = self.get_interval_prediction(y_forward)
        return y_pred


    def draw_graph(self, X, y, file_name, file_type='.png'):
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.xlabel("X0", fontsize=20)
        plt.ylabel("X1", fontsize=20) 
        plt.scatter(X[:,0], X[:,1], s=60, c=y)
        plt.scatter(self.centers[:,0], self.centers[:,1], s=60, c='r')
        for i, r in enumerate(self.sigma):
            draw_circle = plt.Circle(list(self.centers)[i], r, fill=False)
            ax.add_artist(draw_circle)
        plt.savefig(self.root + file_name + file_type)
        plt.close(fig)
    
        
def main(n_samples, n_classes, hidden_dim, epoch_p, lr, std):
    result = f""
    result += f"{n_samples = }\n{n_classes = }\n{hidden_dim = }\n{epoch_p = }\n"
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=42, centers=n_classes, cluster_std=std)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    # X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y_train = np.array([1, 0, 0, 1])
    # X, y = X_train, y_train

    root = f'D:/Applications/vscode/workspace/JMLM/outputs/ns{n_samples}nc{n_classes}hd{hidden_dim}epoch{epoch_p}lr{lr}t{datetime.now().strftime("%Y%m%d%p%I%M%S")}/'

    rbfn = RBFNetwork(hidden_dim=hidden_dim, n_max_center=200, threshold=0.1, epoch=10, epoch_p=epoch_p, lr=lr, root=root)
    rbfn = rbfn.fit(X_train, y_train)
    
    y_train_preds = rbfn.predict(X_train)
    # print(f"{y_train_preds = }")
    train_acc = accuracy_score(y_train, y_train_preds)
    result += f"{train_acc = }\n"
    y_test_preds = rbfn.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_preds)
    result += f"{test_acc = }\n"

    y_vote_preds = rbfn.vote_predict(X_train)
    vote_train_acc = accuracy_score(y_train, y_vote_preds)
    result += f"{vote_train_acc = }\n"

    y_test_vote_preds = rbfn.vote_predict(X_test)
    vote_test_acc = accuracy_score(y_test, y_test_vote_preds)
    result += f"{vote_test_acc = }\n"
    print(result)
    f = open(root + "acc.txt", "w")
    f.write(result)
    f.close()

    rbfn.draw_graph(X, y, f'final_train_acc{train_acc}_test_acc{test_acc}')
 
if __name__ == "__main__":
    n_samples, n_classes, hidden_dim, epoch_p, lr, std = 300, 3, 8, 2000, 0.1, 1
    main(n_samples, n_classes, hidden_dim, epoch_p, lr, std)
    # for i in range(2, 10):
    #     main(n_samples, n_classes, i, epoch_p, lr, std)

    # for i in range(3, 10):
    #     for j in range(i-1, 10):
    #         main(n_samples, i, j, epoch_p, lr, std)

    # for s in range(1, 5):
    #     main(n_samples, n_classes, 5, epoch_p, lr, s)