from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from load_data import load_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import logging
import time
import tqdm

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



class JMLM():
    def __init__(self) -> None:
        self.points = []
        self.d_points = []
        self.jacobians = []


    def clustering(self, X_train, y_train, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters).fit(X_train)
        X_closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_train)
        replaced_centers = X_train[X_closest]
        d_points = y_train[X_closest]
        return replaced_centers, d_points

    def computing_jacobian(self, X, y, replaced_centers, d_points):
        clusters, _ = pairwise_distances_argmin_min(X, replaced_centers)
        jacobians = []
        for ci in range(len(replaced_centers)):
            X_ci = X[clusters==ci]
            y_ci = y[clusters==ci]
            D_p = X_ci - replaced_centers[ci]
            D_F = y_ci - d_points[ci]
            jacobian = np.transpose(D_F) @ D_p @ (np.transpose(np.linalg.pinv(np.transpose(D_p) @ D_p)))
            jacobians.append(jacobian)
            
        return jacobians

    def deep_train(self, X, y, max_points=10, threshold=0.1):
        training_data = [[X, y]]
        layer = 0
        min_mse = float('inf')
        while len(self.points) < max_points and training_data:
            logging.info(f"======Layer {layer}======")
            next_data = []
            layer_max_node = 3
            node_threshold = threshold
            
            for data in training_data:
                if len(data[0]):
                    mse, partial_mse, points, d_points, jacobians = self.train(data[0], data[1], max_points=layer_max_node, threshold=node_threshold, deep=True)
                    if mse < threshold:
                        min_mse = mse
                        break
                    logging.info(f"Individual MSE: {np.array(partial_mse)}")
                    for ci, enough in enumerate(np.array(partial_mse) < threshold):
                        if not enough :
                            clusters, _ = pairwise_distances_argmin_min(data[0], points)
                            new_X = data[0][clusters==ci]
                            next_data.append([new_X, data[1][clusters==ci]])
            if min_mse < threshold: break
            training_data = next_data
            layer += 1


    def train(self, X, y, max_points=10, threshold=0.1, deep=False):
        min_mse = float('inf')
        target_partial_mse = []
        target_points = []
        target_d_points = []
        target_jacobians = []
        K = 1 if not deep else 2
        
        for n_point in tqdm.tqdm(range(K, max_points + 1)):
            logging.info(f"num of max point: {n_point}")
            points, d_points = self.clustering(X, y, n_point)
            jacobians = self.computing_jacobian(X, y, points, d_points)
            mse, partial_mse = self.get_mse(X, y, points, d_points, jacobians, deep)

            logging.info(f"MSE: {mse}")
            if mse > min_mse: continue
            min_mse = mse
            target_points = points
            target_d_points = d_points
            target_jacobians = jacobians
            target_partial_mse = partial_mse
            if min_mse < threshold: break
        
        self.points += target_points.tolist()
        self.d_points += target_d_points.tolist()
        self.jacobians += target_jacobians
        logging.info(f"num of center: {len(self.points)}")
        if deep:
            return min_mse, target_partial_mse, target_points, target_d_points, target_jacobians


    def get_mse(self, X, y, points, d_points, jacobians, deep):
        y_preds = self.forward(X, train=True, replaced_centers=points, d_points=d_points, jacobians=jacobians)
        mse = mean_squared_error(np.array(y), y_preds)
        partial_mse = []
        if deep:
            clusters, _ = pairwise_distances_argmin_min(X, points)
            for ci in range(len(points)):
                X_ci = X[clusters==ci]
                y_ci = y[clusters==ci]
                y_preds = self.forward(X_ci, train=True, replaced_centers=points[ci].reshape(1, -1), d_points=d_points[ci].reshape(1, -1), jacobians=[jacobians[ci]])
                partial_mse.append(mean_squared_error(np.array(y_ci), y_preds))
        return mse, partial_mse

    
    def forward(self, X, train=False, replaced_centers=None, d_points=None, jacobians=None):
        replaced_centers = replaced_centers if train else np.array(self.points)
        d_points = d_points if train else np.array(self.d_points)
        jacobians = jacobians if train else self.jacobians
        clusters, _ = pairwise_distances_argmin_min(X, replaced_centers)
    
        y_preds = []
        for xi, ci in enumerate(clusters):
            y_pred = d_points[ci] + jacobians[ci] @ (X[xi] - replaced_centers[ci])
            y_preds.append(y_pred)

        return np.stack(y_preds)

        
    def predict(self, X):
        return self.forward(X).argmax(axis=1)




if __name__ == '__main__':
    logging.basicConfig(filename='new_JMLM.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %p %I:%M:%S')
    logging.info(f"===============Execution: JMLM===============")
    dataset = "mnist"
    X_train, X_test, y_train, y_test = load_data(dataset)
    logging.info(f"Dataset: {dataset}")

    n_pca = int(X_train.shape[1]*0.8)

    pca = PCA(n_components=n_pca)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    clf = LinearDiscriminantAnalysis()
    X_train = clf.fit_transform(X_train, y_train.argmax(axis=1))
    X_test = clf.transform(X_test)

    start_time = time.time()
    jmlm = JMLM()
    jmlm.deep_train(X_train, y_train, max_points=100, threshold=0.001)
    # jmlm.train(X_train, y_train, max_points=50, threshold=0.01)

    y_pred = jmlm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f'Prediction Accuracy: {acc}')
    logging.info("--- %s seconds ---" % (time.time() - start_time))
    
    