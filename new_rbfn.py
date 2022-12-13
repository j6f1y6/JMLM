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
from sklearn.decomposition import PCA

class RBFNetwork():
    def __init__(self, hidden_dim=2, kmeans_iter=20, threshold=0.1, n_max_center=100, lr=0.5, epoch=2000, epoch_p=100, kernel: Kernel = Gaussian(), root=f'./outputs/{datetime.now().strftime("%Y%m%d%p%I%M%S")}/', data_type=0, update_type=0, pca=None) -> None:
        self.hidden_dim = hidden_dim
        self.kmeans_iter = kmeans_iter
        self.threshold = threshold
        self.lr = lr
        self.epoch = epoch
        self.kernel = kernel
        self.n_max_center = n_max_center
        self.root = root
        self.epoch_p = epoch_p
        self.data_type = data_type
        self.update_type = update_type
        self.pca=pca


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
        y_pred = self.get_interval_prediction(y_forward)
        # output_grad = y - y_pred
        output_grad = y_pred - y
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
        for ci in range(len(self.centers)):
            X_c = X if self.data_type == 2 else X[clusters==ci]
            y_c = y if self.data_type == 2 else y[clusters==ci]
            if self.update_type == 0:
                all_index = np.arange(len(X_c))
                np.random.shuffle(all_index)
                for ri in all_index:
                    
                    xi = X_c[ri].reshape(1, -1)
                    yi = y_c[ri]
                    y_forward = self.kernel.compute(xi, self.centers[ci], self.sigma[ci])
                    self.backward(xi, y_forward, yi, ci)


            elif self.update_type == 1:
                    y_forward = self.cluster_forward(ci, X_c)
                    self.cluster_backward(ci, X_c, y_forward, y_c)

        # 每群有outputdim個std去更新
        # clusters, _ = pairwise_distances_argmin_min(X, self.centers)
        # for ci in range(len(self.centers)):
        #     y_forward = self.cluster_forward(ci, X[clusters==ci])
        #     for class_idx in range(self.output_dim):
        #         self.cluster_backward(ci, class_idx, X[clusters==ci], y_forward[class_idx], y[clusters==ci][class_idx])

    def get_mse(self, X, y, clusters):
        # mse = float('inf')
        y_forward = self.predict(X)
        mse = mean_squared_error(y, y_forward)

        mse_p = []
        for ci in range(len(self.centers)):
            xci = X[clusters==ci]
            yci = y[clusters==ci]
            if xci.shape[0] == 0:
                mse_p.append(0)
                continue
            yci_forward = self.predict(xci)
            mse_p.append(mean_squared_error(yci, yci_forward))
        return mse, mse_p


    def train(self, X, y):
        self.initialize_parameters(X, y)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
        X_train, y_train = X, y
        train_data = [[X_train, y_train]]
        node_index = [0]
        layer = 0

        while train_data and self.n_max_center > self.centers.shape[0]:
            print(f"=============={layer = }==============")
            next_data = []
            print(f"{len(train_data) = }")
            for data, ni in zip(train_data, node_index):
                X_part, y_part = data[0], data[1]
                # print(f"{X_part.shape = }, {y_part.shape = }")
                
                if X_part.shape[0] < self.hidden_dim: continue
                new_centers, new_sigma = self.add_center(X_part, y_part)
            
                self.centers = np.concatenate((self.centers, new_centers)) if self.centers.size > 0 else new_centers
                self.sigma += new_sigma

            clusters, _ = pairwise_distances_argmin_min(X_train, self.centers)
            
            self.file_name = 0
            for i in tqdm.tqdm(range(self.epoch_p)):
                if self.data_type == 1:
                    clusters, _ = pairwise_distances_argmin_min(X_train, self.centers)
                elif self.data_type == 2:
                    clusters = None
                
                # if i % 100 == 0:
                if True:
                    if self.pca is not None:
                        X_pca = self.pca.transform(X)
                        self.draw_graph(X_pca, y, str(i)) 
                        pass
                    else:
                        self.draw_graph(X, y, str(i)) 

                self.train_one_epoch(X_train, y_train, clusters=clusters)

            # mse, mse_p = self.get_mse(X_train, y_train, clusters)
            # print(f"{mse = }")

            # if(len(self.centers) > self.n_max_center): break

            # threshold = 0.3

            # for ci in range(len(self.centers)):
            #     xci = X_train[clusters==ci]
            #     yci = y_train[clusters==ci]
            #     if xci.size == 0: continue
            #     ycif = self.predict(xci)
            #     yciacc = accuracy_score(yci, ycif)
            #     element = np.unique(np.array(yci), axis=0)
            #     n_X = xci.shape[0]
            #     print(f"{self.centers[ci]}, {element = }, {yciacc = }, {mse_p[ci] = }, {n_X = }")
            #     if mse_p[ci] > threshold:
            #         next_data.append([xci, yci])
            # print("======================")

            print(f"current n centers: {len(self.centers)}")
            train_data = next_data
            layer += 1

    def fit(self, X, y):
        self.train(X, y)
        return self

    def forward(self, X, t_centers=None, t_sigma=None, clusters=None):
        centers = t_centers if t_centers is not None else self.centers
        sigma = t_sigma if t_sigma is not None else self.sigma
        y_preds = []
        if clusters is None:
            clusters, _ = pairwise_distances_argmin_min(X, centers)
        # print(clusters)
 
        for idx, x in enumerate(X):
            rbf = self.kernel.compute(x, centers[clusters[idx]], sigma[clusters[idx]])
            y_preds.append(rbf)
        return y_preds

    def backward(self, x, y_forward, y, ci):
        center = self.centers[ci]
        sigma = self.sigma[ci]
        y_pred = self.get_interval_prediction([y_forward])
        grad = y - y_pred
        d_center = grad * y_forward * (x - center) / (sigma ** 2)
        d_sigma = grad * y_forward * np.linalg.norm(x - center)**2 / (sigma ** 3)
        self.centers[ci] = center + self.lr * d_center
        self.sigma[ci] = sigma + self.lr * d_sigma
        

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
        centers = self.centers if self.pca is None else self.pca.transform(self.centers)
        plt.scatter(centers[:,0], centers[:,1], s=60, c='r')
        for sigma, center in zip(self.sigma, centers):
            draw_circle = plt.Circle(center, sigma, fill=False)
            ax.add_artist(draw_circle)
        # for i, r in enumerate(self.sigma):
        #     # draw_circle = plt.Circle(list(self.centers)[i], r, fill=False)
        #     draw_circle = plt.Circle(self.centers[i], r, fill=False)
        #     ax.add_artist(draw_circle)
        plt.savefig(self.root + file_name + file_type)
        plt.close(fig)
        
    
        
def main(n_samples, n_classes, hidden_dim, epoch_p, lr, std, data_type, update_type, root):
    data_type_list = {0: "fixed", 1: "updated", 2: "all"}
    update_type_list = {0: "one", 1: "cluster"}
    result = f""
    result += f"{data_type_list[data_type] = }\n{update_type_list[update_type] = }\n{n_samples = }\n{n_classes = }\n{hidden_dim = }\n{epoch_p = }\n"
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=42, centers=n_classes, cluster_std=std)
    draw_X, draw_y = X, y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    dataset = 'ICU'
    X_train, X_test, y_train, y_test = load_data(dataset, onehot=False)
    pca = PCA(n_components=2)
    pca.fit(X_train)
    # X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y_train = np.array([1, 0, 0, 1])
    # X, y = X_train, y_train

    


    rbfn = RBFNetwork(hidden_dim=hidden_dim, n_max_center=20, threshold=0.1, epoch=10, epoch_p=epoch_p, lr=lr, root=root, data_type=data_type, update_type=update_type, pca=pca)
    # rbfn = rbfn.fit(X, y)
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

    # rbfn.draw_graph(draw_X, draw_y, f'final')
    rbfn.draw_graph(draw_X, draw_y, f'final')
 
if __name__ == "__main__":
    n_samples, n_classes, hidden_dim, epoch_p, lr, std = 1000, 6, 5 , 100, 0.5, 1
    # data_type_list = {0: "fixed", 1: "updated", 2: "all"}
    # update_type_list = {0: "one", 1: "cluster"}
    data_type = 1
    update_type = 0
    root = f'D:/Applications/vscode/workspace/JMLM/outputs/'
    # final_dir = f'ns{n_samples}nc{n_classes}hd{hidden_dim}epoch{epoch_p}lr{lr}t{datetime.now().strftime("%Y%m%d%p%I%M%S")}/'
    # main(n_samples, n_classes, hidden_dim, epoch_p, lr, std, data_type, update_type, root + final_dir)

    tmp_dir = root + f"ICU/"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    for n_center in range(1, 20):
        tmp_center_dir = tmp_dir + f'center_{n_center}/'
        if not os.path.exists(tmp_center_dir):
            os.makedirs(tmp_center_dir)
        for epoch in range(500, 1001, 500):
            final_dir = f'hd{n_center}epoch{epoch}lr{lr}t{datetime.now().strftime("%Y%m%d%p%I%M%S")}/'
            main(n_samples, n_classes, n_center, epoch, lr, std, data_type, update_type, tmp_center_dir + final_dir)
    # for n_class in range(3, 10):
    #     tmp_dir = root + f'class_{n_class}/'
    #     if not os.path.exists(tmp_dir):
    #         os.makedirs(tmp_dir)
    #     for n_center in range(1, 20):
    #         tmp_center_dir = tmp_dir + f'center_{n_center}/'
    #         if not os.path.exists(tmp_center_dir):
    #             os.makedirs(tmp_center_dir)
    #         for epoch in range(500, 1001, 500):
    #             final_dir = f'ns{n_samples}nc{n_class}hd{n_center}epoch{epoch}lr{lr}t{datetime.now().strftime("%Y%m%d%p%I%M%S")}/'
    #             main(n_samples, n_class, n_center, epoch, lr, std, data_type, update_type, tmp_center_dir + final_dir)

    