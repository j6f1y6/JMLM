import os
import time
import tqdm
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, accuracy_score, mean_squared_error

from load_data import load_data

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
logging.getLogger('matplotlib.font_manager').disabled = True


class JMLM():
    def __init__(self) -> None:
        self.points = []
        self.d_points = []
        self.jacobians = []

    def train(self, X, y, X_kmeans, y_kmeans, max_points=10, threshold=0.1, deep=False, kmeans_iter=10, n_current_node=0):
        X, X_valid, y, y_valid = train_test_split(X, y, test_size=0.2)
        min_mse = float('inf')
        target_points = []
        target_d_points = []
        target_jacobians = []

        K = 1

        for n_point in tqdm.tqdm(range(K, max_points + 1)):
            if len(X_kmeans) < n_point: break
            logging.info(f"current num of max point: {n_point}")
            mse_list = []
            for _ in range(kmeans_iter):
                points, d_points = self.clustering(X_kmeans, y_kmeans, n_point)
                jacobians = self.computing_jacobian(X, y, points, d_points)
                # mse = self.get_mse(X, y, points, d_points, jacobians)
                mse, partial_mse = self.get_mse(X_valid, y_valid, points, d_points, jacobians)
                mse_list.append(mse)

                logging.info(f"{n_point = } MSE: {mse}")
                if mse > min_mse: continue
                min_mse = mse

                target_points = points
                target_d_points = d_points
                target_jacobians = jacobians
                if min_mse < threshold: break

        logging.info(f"{target_points.shape = }")

        self.points += target_points.tolist()
        self.d_points += target_d_points.tolist()
        self.jacobians += target_jacobians
        logging.info(f"final num of center: {len(self.points)}")
        
  
    def deep_train(self, X, y, max_points=10, layer_max_node=3, threshold=0.1, kmeans_iter=20):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
        
        training_data = [[X_train, y_train]]
        layer = 0
        
        while training_data and len(self.points) < max_points:
            logging.info(f"======Layer {layer}======")
            next_data = []
            for data in training_data:
                X_train = data[0]
                y_train = data[1]
                if len(X_train) < layer_max_node: continue
                points, d_points, jacobians, mse, partial_mse = None, None, None, None, None
                tmp_min_mse = float('inf')
                K = 2
                for layer_node in tqdm.tqdm(range(K, layer_max_node + 1)):
                    for _ in range(kmeans_iter):
                        tmp_points, tmp_d_points = self.clustering(X_train, y_train, layer_node)
                        tmp_jacobians = self.computing_jacobian(X, y, tmp_points, tmp_d_points)
                        tmp_mse, tmp_partial_mse = self.get_mse(X_valid, y_valid, tmp_points, tmp_d_points, tmp_jacobians, deep, X_train, y_train)
                        if tmp_mse <= tmp_min_mse:
                            tmp_min_mse = tmp_mse
                            points, d_points, jacobians, mse, partial_mse = tmp_points, tmp_d_points, tmp_jacobians, tmp_mse, tmp_partial_mse
                if len(self.points) > max_points: break

                self.points += points.tolist()
                self.d_points += d_points.tolist()
                self.jacobians += jacobians
                logging.info(f"{mse = }")
                logging.info(f"{partial_mse = }")
                logging.info(f"{len(self.points) = }")
                clusters, _ = pairwise_distances_argmin_min(X_train, points)
                for ci, p_mse in enumerate(partial_mse):
                    enough = p_mse < threshold
                    if not enough:
                        next_data.append([X_train[clusters==ci], y_train[clusters==ci]])
                
                if mse < threshold:
                    next_data = []
                    break

                
            layer += 1
            training_data = next_data

        

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
            jacobian = np.transpose(D_F) @ D_p @ (np.transpose(np.linalg.pinv(np.transpose(D_p) @ D_p)))
            jacobians.append(jacobian)
        return jacobians


    def get_mse(self, X, y, points, d_points, jacobians, deep=False, X_partial=None, y_partial=None):
        y_forward = self.forward(X, train=True, replaced_centers=np.array(self.points + list(points)), d_points=np.array(self.d_points + list(d_points)), jacobians=self.jacobians + jacobians)
        mse = mean_squared_error(np.array(y), y_forward)

        partial_mse = []
        if deep:
            clusters, _ = pairwise_distances_argmin_min(X_partial, points)
            for ci in range(len(points)):
                X_ci = X_partial[clusters==ci]
                y_ci = y_partial[clusters==ci]
                if len(X_ci) <= 0: continue
                y_forward = self.forward(X_ci, train=True, replaced_centers=points[ci].reshape(1, -1), d_points=d_points[ci].reshape(1, -1), jacobians=[jacobians[ci]])
                partial_mse.append(mean_squared_error(np.array(y_ci), y_forward))
        return mse, partial_mse

    
    def forward(self, X, train=False, replaced_centers=None, d_points=None, jacobians=None):
        replaced_centers = replaced_centers if train else np.array(self.points)
        d_points = d_points if train else np.array(self.d_points)
        jacobians = jacobians if train else self.jacobians
        clusters, _ = pairwise_distances_argmin_min(X, replaced_centers)
    
        y_forwards = []
        for xi, ci in enumerate(clusters):
            y_forward = d_points[ci] + jacobians[ci] @ (X[xi] - replaced_centers[ci])
            y_forwards.append(y_forward)

        return np.stack(y_forwards)

    def predict(self, X):
        return self.forward(X)

    
def main(dataset="asmpt_train", n_max_node=5, threshold=0.01, kmeans_iter=10, asmpt_target=5, deep=False, layer_max_node=5):
    logging.basicConfig(filename='reg_JMLM_deep.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %p %I:%M:%S')
    logging.info(f"===============Execution: JMLM===============")

    X_train, X_test, y_train, y_test = load_data(dataset, onehot=False, normalization=False, asmpt_target=asmpt_target, classification=False)

    print(f"Running JMLM...\nDataset: {dataset}")
    
    start_time = time.time()
    jmlm = JMLM()
    if deep:
        # jmlm.train(X_train, y_train, X_train, y_train, n_max_node, threshold, False, kmeans_iter)
        jmlm.deep_train(X_train, y_train, max_points=n_max_node, layer_max_node=layer_max_node, threshold=threshold, kmeans_iter=kmeans_iter)
    else:
        jmlm.train(X_train, y_train, X_train, y_train, n_max_node, threshold, False, kmeans_iter)
    


    y_train_pred = jmlm.predict(X_train)
    y_test_pred = jmlm.predict(X_test)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")

    return train_mse, test_mse


if __name__ == '__main__':
    # deep = True
    # n_max_node = 100
    # kmeans_iter = 10
    # target = 0
    # layer_max_node = 5
    # train_mse, test_mse = main(
    #                             dataset="asmpt_train", 
    #                             n_max_node=n_max_node, 
    #                             threshold=0.5, 
    #                             kmeans_iter=kmeans_iter, 
    #                             asmpt_target=target,
    #                             deep=deep,
    #                             layer_max_node=layer_max_node
    #                         )

    
    ## Deep JMLM
    deep = True
    n_centers = list(range(5, 101, 5))
    kmeans_iter = 10
    targets = list(range(0, 6))
    layer_max_node = 5
    for target in targets:
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        train_mse_list = []
        test_mse_list = []
        for n_max_node in n_centers:
            train_mse, test_mse = main(
                                dataset="asmpt_train", 
                                n_max_node=n_max_node, 
                                threshold=0.5, 
                                kmeans_iter=kmeans_iter, 
                                asmpt_target=target,
                                deep=deep,
                                layer_max_node=layer_max_node
                            )
            train_mse_list.append(train_mse)
            test_mse_list.append(test_mse)
        min_idx = test_mse_list.index(min(test_mse_list))
        train_min = "{:.4f}".format(train_mse_list[min_idx])
        test_min = "{:.4f}".format(test_mse_list[min_idx])
        ax.set_title(f"kmeans_iter: {kmeans_iter}\nnumber of centers:{(min_idx+1*5)}\nTrain MSE: {train_min}, Test MSE: {test_min}")
        ax.plot(n_centers, train_mse_list, label=f'Train', marker='o')
        ax.plot(n_centers, test_mse_list, label=f'Test', marker='s', c='r')
        for x, y in enumerate(train_mse_list):
            label = "{:.4f}".format(y)
            ax.annotate(label, # this is the text
                        ((x+1)*5,y), # these are the coordinates to position the label
                        textcoords="offset points", # how to position the text
                        xytext=(0,10), # distance from text to points (x,y)
                        size=8,
                        ha='center') # horizontal alignment can be left, right or center
            
        for x, y in enumerate(test_mse_list):
            label = "{:.4f}".format(y)
            ax.annotate(label, # this is the text
                        ((x+1)*5,y), # these are the coordinates to position the label
                        textcoords="offset points", # how to position the text
                        xytext=(0,10), # distance from text to points (x,y)
                        size=8,
                        ha='center') # horizontal alignment can be left, right or center
        ax.legend(loc='best')
        ax.grid('on')
        ax.set_xlabel(r"Number of Centers")
        ax.set_ylabel(r"$MSE$")
        fig.canvas.manager.set_window_title(f"target {target}")
        plt.savefig(f"D:/Applications/vscode/workspace/JMLM/datasets/asmpt/outputs/deep_target_{target}.png")
        plt.close(fig)
                
            
    ## Normal JMLM
    # targets = list(range(0, 6))
    # n_centers = list(range(1, 21))
    # kmeans_iters = list(range(10, 21, 10))
    # # targets = list(range(0, 2))
    # # n_centers = list(range(1, 3))
    # # kmeans_iters = list(range(10, 21, 10))

    # for target in targets:
    #     fig, ax = plt.subplots(1, 2, figsize=(16, 10))
    #     for kmeans_iter in kmeans_iters:
    #         train_mse_list = []
    #         test_mse_list = []
    #         for n_max_node in n_centers:
    #             train_mse, test_mse = main(
    #                                         dataset="asmpt_train", 
    #                                         n_max_node=n_max_node, 
    #                                         threshold=0.01, 
    #                                         kmeans_iter=kmeans_iter, 
    #                                         asmpt_target=target,
    #                                         deep=deep
    #                                     )
    #             train_mse_list.append(train_mse)
    #             test_mse_list.append(test_mse)
    
    #         axi = int(kmeans_iter/10 - 1)
    #         min_idx = test_mse_list.index(min(test_mse_list))
    #         train_min = "{:.4f}".format(train_mse_list[min_idx])
    #         test_min = "{:.4f}".format(test_mse_list[min_idx])
    #         ax[axi].set_title(f"kmeans_iter: {kmeans_iter}\nnumber of centers:{min_idx+1}\nTrain MSE: {train_min}, Test MSE: {test_min}")
    #         ax[axi].plot(n_centers, train_mse_list, label=f'Train', marker='o')
    #         ax[axi].plot(n_centers, test_mse_list, label=f'Test', marker='s', c='r')
    #         for x, y in enumerate(train_mse_list):
    #             label = "{:.4f}".format(y)
    #             ax[axi].annotate(label, # this is the text
    #                         (x+1,y), # these are the coordinates to position the label
    #                         textcoords="offset points", # how to position the text
    #                         xytext=(0,10), # distance from text to points (x,y)
    #                         ha='center') # horizontal alignment can be left, right or center
    #         for x, y in enumerate(test_mse_list):
    #             label = "{:.4f}".format(y)
    #             ax[axi].annotate(label, # this is the text
    #                         (x+1,y), # these are the coordinates to position the label
    #                         textcoords="offset points", # how to position the text
    #                         xytext=(0,10), # distance from text to points (x,y)
    #                         ha='center') # horizontal alignment can be left, right or center
    #         ax[axi].legend(loc='best')
    #         ax[axi].grid('on')
    #         ax[axi].set_xlabel(r"Number of Centers")
    #         ax[axi].set_ylabel(r"$MSE$")
    #     fig.canvas.manager.set_window_title(f"target {target}")
    #     plt.savefig(f"D:/Applications/vscode/workspace/JMLM/datasets/asmpt/outputs/target_{target}.png")
    #     plt.close(fig)
        