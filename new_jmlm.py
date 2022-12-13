import os
import time
from psutil import NORMAL_PRIORITY_CLASS
import tqdm
import wandb
import logging
import datetime
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances_argmin_min, accuracy_score, mean_squared_error

import jmlm as old
from load_data import load_data

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
logging.getLogger('matplotlib.font_manager').disabled = True


class JMLM():
    def __init__(self) -> None:
        self.points = []
        self.d_points = []
        self.jacobians = []
        self.graph_data = []
        self.dot = Digraph(comment='Tree')


    def view(self):
        for node in self.graph_data:
            self.dot.node(node["index"], node["detail"])
            if node["parent"]:
                self.dot.edge(node["parent"], node["index"])
        self.dot.view()


    def jmlm_train(self, X, y, max_points=10, threshold=0.1, deep=False, to_max=False, kmeans_iter=5):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
        self.X_valid = X_valid
        self.y_valid  = y_valid
        return self.train(X_train, y_train, X_train, y_train, max_points, threshold, deep, to_max, kmeans_iter)


    def deep_train(self, X, y, max_points=10, layer_max_node=3, threshold=0.1, kmeans_iter=20, fixed=False):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.y_labels = np.unique(np.array(y), axis=0)
        self.y_labels = np.array(sorted(self.y_labels, key=lambda e: e.argmax()))

        if len(self.graph_data) == 0:
            data_info = f"n_feature: {X.shape[1]}\nsamples: {X.shape[0]}"
            self.graph_data.append({"index" : str(0), "detail" : data_info, "parent" : None})
        
        training_data = [[X_train, y_train]]
        node_index = [0]
        min_mse = float('inf')
        layer = 0
        
        while training_data and len(self.points) < max_points:
            logging.info(f"======Layer {layer}======")
            next_data = []
            next_node_index = []
            for data, ni in zip(training_data, node_index):
                X_train = data[0]
                y_train = data[1]
                if len(X_train) < layer_max_node: continue
                points, d_points, jacobians, mse, partial_mse = None, None, None, None, None
                tmp_min_mse = float('inf')
                K = layer_max_node if fixed else 2
                for layer_node in range(K, layer_max_node + 1):
                    for _ in range(kmeans_iter):
                        tmp_points, tmp_d_points = self.clustering(X_train, y_train, layer_node)
                        tmp_jacobians = self.computing_jacobian(X, y, tmp_points, tmp_d_points)
                        tmp_mse, tmp_partial_mse, _, _ = self.get_mse(X, y, tmp_points, tmp_d_points, tmp_jacobians, True, X_train, y_train)
                        if tmp_mse <= tmp_min_mse:
                            tmp_min_mse = tmp_mse
                            points, d_points, jacobians, mse, partial_mse = tmp_points, tmp_d_points, tmp_jacobians, tmp_mse, tmp_partial_mse
                # if mse > min_mse: continue
                # min_mse = mse
                if len(self.points) > max_points: break

                self.points += points.tolist()
                self.d_points += d_points.tolist()
                self.jacobians += jacobians
                logging.info(mse)
                logging.info(partial_mse)
                clusters, _ = pairwise_distances_argmin_min(X_train, points)
                n_current = len(self.graph_data)
                for ci in range(len(points)):
                    center_info = list(np.around(np.array(points[ci]), 2))
                    center_mse = partial_mse[ci]
                    n_sample = X_train[clusters==ci].shape[0]
                    node_detail = f"center: {center_info}\nmse: {center_mse}\nsamples: {n_sample}"
                    self.graph_data.append({"index" : str(n_current), "detail" : node_detail, "parent" : str(ni)})
                    n_current += 1

                for ci, p_mse in enumerate(partial_mse):
                    enough = p_mse < threshold
                    if not enough:
                        next_data.append([X_train[clusters==ci], y_train[clusters==ci]])
                        next_node_index.append(len(self.graph_data) - len(partial_mse) + ci)
                
                if mse < threshold:
                    next_data = []
                    break

                
            layer += 1
            training_data = next_data
            node_index = next_node_index
            

        


    def train(self, X, y, X_kmeans, y_kmeans, max_points=10, threshold=0.1, deep=False, to_max=False, kmeans_iter=10, n_current_node=0):
        self.y_labels = np.unique(np.array(y), axis=0)
        self.y_labels = np.array(sorted(self.y_labels, key=lambda e: e.argmax()))
        min_mse = float('inf')
        max_acc = 0
        target_partial_mse = []
        target_points = []
        target_d_points = []
        target_jacobians = []
        all_acc_list = []
        all_mse_list = []
        all_knn_list = []

        K = max_points if to_max or deep else 1

        for n_point in tqdm.tqdm(range(K, max_points + 1)):
            if len(X_kmeans) < n_point: break
            logging.info(f"current num of max point: {n_point}")
            acc_list = []
            mse_list = []
            knn_list = []
            for _ in range(kmeans_iter):
                points, d_points = self.clustering(X_kmeans, y_kmeans, n_point)
                jacobians = self.computing_jacobian(X, y, points, d_points)
                if deep:
                    mse, partial_mse, acc, knn_acc = self.get_mse(X, y, points, d_points, jacobians, deep, X_kmeans, y_kmeans)
                else:
                    mse, partial_mse, acc, knn_acc = self.get_mse(self.X_valid, self.y_valid, points, d_points, jacobians, deep)
                acc_list.append(acc)
                mse_list.append(mse)
                knn_list.append(knn_acc)

                if deep:
                    logging.info(f"{n_point = } MSE: {mse}")
                    if mse > min_mse: continue
                    min_mse = mse
                else:
                    logging.info(f"{n_point = } ACC: {acc}")
                    if acc < max_acc: continue
                    max_acc = acc

                target_points = points
                target_d_points = d_points
                target_jacobians = jacobians
                target_partial_mse = partial_mse
                if min_mse < threshold: break
            ki = np.array(acc_list).argmax(axis=0)
            all_acc_list.append(acc_list[ki])
            all_mse_list.append(mse_list[ki])
            all_knn_list.append(knn_list[ki])
        
        logging.info(f"{target_points.shape = }")

        self.points += target_points.tolist()
        self.d_points += target_d_points.tolist()
        self.jacobians += target_jacobians
        logging.info(f"final num of center: {len(self.points)}")
        if not deep:
            mse, target_partial_mse, _, _ = self.get_mse(X, y, np.array(self.points), np.array(self.d_points), self.jacobians, True, X, y)
        
        if len(self.graph_data) == 0:
            data_info = f"n_feature: {X.shape[1]}\nmse: {mse}\nsamples: {X.shape[0]}"
            self.graph_data.append({"index" : str(n_current_node), "detail" : data_info, "parent" : None})

        n_parent = str(n_current_node)
        n_current = len(self.graph_data)
        clusters, _ = pairwise_distances_argmin_min(X_kmeans, target_points)
        print(X_kmeans.shape)
        for ci in range(len(target_points)):
            X_ci = X_kmeans[clusters==ci]
            node_detail = f"center: {list(np.around(np.array(target_points[ci]), 2))}\nmse: {target_partial_mse[ci]}\nsamples: {X_ci.shape[0]}"
            self.graph_data.append({"index" : str(n_current), "detail" : node_detail, "parent" : n_parent})
            n_current += 1

        if deep:
            return min_mse, target_partial_mse, target_points
        return all_acc_list, all_knn_list, all_mse_list


    def clustering(self, X_train, y_train, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++').fit(X_train)
        X_closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_train)

        replaced_centers = X_train[X_closest]
        d_points = y_train[X_closest]
        return replaced_centers, d_points


    def computing_jacobian(self, X, y, replaced_centers, d_points):
        jacobians = []
        for ci in range(len(replaced_centers)):
            D_p = X - replaced_centers[ci]
            D_F = y - d_points[ci]

            # nxN @ Nxm => nxm
            # J_p = np.linalg.pinv(D_p) @ D_F
            # jacobians.append(np.transpose(J_p))

            jacobian = np.transpose(D_F) @ D_p @ (np.transpose(np.linalg.pinv(np.transpose(D_p) @ D_p)))
            jacobians.append(jacobian)
            
        return jacobians


    def get_mse(self, X, y, points, d_points, jacobians, deep, X_partial=None, y_partial=None):
        # y_forward, knn_forward = self.forward(X, train=True, replaced_centers=points, d_points=d_points, jacobians=jacobians)
        y_forward, knn_forward = self.forward(X, train=True, replaced_centers=np.array(self.points + list(points)), d_points=np.array(self.d_points + list(d_points)), jacobians=self.jacobians + jacobians)
        mse = mean_squared_error(np.array(y), y_forward)
        y_acc_pred, _ = pairwise_distances_argmin_min(y_forward, self.y_labels)
        acc = accuracy_score(y.argmax(axis=1), y_acc_pred)
        knn_acc = accuracy_score(y, knn_forward)

        partial_mse = []
        if deep:
            clusters, _ = pairwise_distances_argmin_min(X_partial, points)
            for ci in range(len(points)):
                X_ci = X_partial[clusters==ci]
                y_ci = y_partial[clusters==ci]
                if len(X_ci) <= 0: continue
                y_forward, _ = self.forward(X_ci, train=True, replaced_centers=points[ci].reshape(1, -1), d_points=d_points[ci].reshape(1, -1), jacobians=[jacobians[ci]])
                partial_mse.append(mean_squared_error(np.array(y_ci), y_forward))
        
        return mse, partial_mse, acc, knn_acc

    
    def forward(self, X, train=False, replaced_centers=None, d_points=None, jacobians=None):
        replaced_centers = replaced_centers if train else np.array(self.points)
        d_points = d_points if train else np.array(self.d_points)
        jacobians = jacobians if train else self.jacobians
        clusters, _ = pairwise_distances_argmin_min(X, replaced_centers)
    
        y_preds = []
        knn = []
        for xi, ci in enumerate(clusters):
            y_pred = d_points[ci] + jacobians[ci] @ (X[xi] - replaced_centers[ci])
            y_preds.append(y_pred)
            knn.append(d_points[ci])

        return np.stack(y_preds), np.stack(knn)

    def predict(self, X):
        y_forward, knn = self.forward(X)
        y_pred, _ = pairwise_distances_argmin_min(y_forward, self.y_labels)
        return y_pred, knn
        # return y_forward.argmax(axis=1), knn


def plot_mse(fig, ax, mse_list):
    ax[1].set_title(f'min MSE: {min(mse_list)}')
    ax[1].plot(mse_list, label='MSE', marker='o')
    ax[1].legend(loc='best')
    ax[1].grid('on')
    ax[1].set_title('MSE')
    return fig, ax
    

def plot_acc(fig, ax, jmlm_list, knn_list):
    ax[0].plot(jmlm_list, label='JMLM', marker='o')
    ax[0].plot(knn_list, label='KNN', marker='x')
    ax[0].legend(loc='best')
    ax[0].grid('on')
    ax[0].set_title(f'Max Valid Accuracy(JMLM max:{max(jmlm_list)})')
    return fig, ax


def data_logger(acc_train, acc_test, knn_acc_train, knn_acc_test, n_max_node, threshold, n_node, dataset, start_time):
    
    logging.info(f"Dataset: {dataset}")
    logging.info(f'ALL KNN Train Prediction Accuracy: {knn_acc_train}')
    logging.info(f'ALL KNN Test Prediction Accuracy: {knn_acc_test}')
    logging.info(f'ALL Train Prediction Accuracy: {acc_train}')
    logging.info(f'ALL Test Prediction Accuracy: {acc_test}')
    logging.info(f"MAX: {n_max_node}, threshold: {threshold}")
    logging.info(f'Number of Node(result): {n_node}')
    logging.info("--- %s seconds ---" % (time.time() - start_time))
    logging.info(f"===============END===============")

def graph_info(dot, acc_train, acc_test, n_max_node, threshold, n_node, dataset, mse, knn_acc, n_layer_node, start_time):
    dot.attr(label=f'Dataset: {dataset}\n'
             f"threshold: {threshold}\n"
             f"MAX points: {n_max_node}\n"
             f"MAX layer points: {n_layer_node}\n"
             f'Number of Node(result): {n_node}\n'
             f'Train Prediction Accuracy: {acc_train}\n'
             f'Test Prediction Accuracy: {acc_test}\n'
             f'Final MSE: {mse}\n'
             f'KNN accuracy: {knn_acc}\n'
             "--- %s seconds ---" % (time.time() - start_time)
             )

    
def main():
    logging.basicConfig(filename='new_JMLM.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %p %I:%M:%S')
    logging.info(f"===============Execution: JMLM===============")
    types = ["JMLM", "Deep", "old"]
    training_type = "JMLM"
    dataset = "ICU"
    n_max_node = 10
    threshold = 0.01
    to_max = False
    kmeans_iter = 10

    # start_n_layer_node = 2 
    max_n_layer_node = 4
    start_n_layer_node = max_n_layer_node
    fixed = False
    
    onehot = False if training_type == types[2] else True
    fig, ax = plt.subplots(1, 2, figsize=(16, 10))

    X_train, X_test, y_train, y_test = load_data(dataset, onehot=onehot)
    # X_train, X_test, y_train, y_test = load_data(dataset, onehot=onehot, pca=0.87, lda=True, noise=True)
    
    print(f"Running JMLM...\nDataset: {dataset}")

    # KNeighbors Classifier
    neigh = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
    y_knn_preds = neigh.predict(X_test)
    y_knn_acc = accuracy_score(y_test, y_knn_preds)


    all_train_acc = 0
    all_test_acc = 0
    for n_layer_node in range(start_n_layer_node, max_n_layer_node + 1):
        start_time = time.time()
        jmlm = JMLM() if training_type != types[2] else old.JMLM()
        acc_list, knn_acc_list, mse_list = [], [], []
        if training_type == types[0]:
            acc_list, knn_acc_list, mse_list = jmlm.jmlm_train(X_train, y_train, max_points=n_max_node, threshold=threshold, to_max=to_max, kmeans_iter=kmeans_iter)
        elif training_type == types[1]:
            jmlm.deep_train(X_train, y_train, max_points=n_max_node, layer_max_node=n_layer_node, threshold=threshold, fixed=fixed)
        elif training_type == types[2]:
            acc_list, knn_acc_list, mse_list = jmlm.train(X_train, y_train, kmeans_iter, n_max_node, X_test, y_test)
            
        y_train_pred, y_train_knn_pred = jmlm.predict(X_train)
        y_test_pred, y_test_knn_pred = jmlm.predict(X_test)
        knn_acc_train = accuracy_score(y_train, y_train_knn_pred)
        knn_acc_test = accuracy_score(y_test, y_test_knn_pred)
        acc_train = accuracy_score(y_train.argmax(axis=1) if onehot else y_train, y_train_pred)
        acc_test = accuracy_score(y_test.argmax(axis=1) if onehot else y_train, y_test_pred)
        train_mse, _, _, _ = jmlm.get_mse(X_train, y_train, jmlm.points, jmlm.d_points, jmlm.jacobians, False)
        all_train_acc += acc_train
        all_test_acc += acc_test

        data_logger(acc_train, acc_test, knn_acc_train, knn_acc_test, n_max_node, threshold, len(jmlm.points), dataset, start_time)
        graph_info(jmlm.dot, acc_train, acc_test, n_max_node, threshold, len(jmlm.points), dataset, train_mse, y_knn_acc, n_layer_node, start_time)
        jmlm.view()
        # draw_graph(X_train, y_train, jmlm)
        if training_type != types[1]:
            fig, ax = plot_acc(fig, ax, acc_list, knn_acc_list)
            fig, ax = plot_mse(fig, ax, mse_list)
            fig.suptitle(f'Dataset: {dataset}\n')
            fig.savefig(dataset + "_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".png")
            plt.show()
    # logging.info(f'ALL Train Prediction Accuracy: {all_train_acc / ((max_n_layer_node - start_n_layer_node + 1)*2)}')
    # logging.info(f'ALL Test Prediction Accuracy: {all_test_acc / ((max_n_layer_node - start_n_layer_node + 1)*2)}')
        

if __name__ == '__main__':
    main()