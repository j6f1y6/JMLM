from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from load_data import load_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import logging
import time


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
        # while(len(self.points) < max_points):
        for data in training_data:
            mse, partial_mse, points, d_points, jacobians = self.train(data[0], data[1], max_points=3, threshold=threshold, deep=True)
            logging.info(np.array(partial_mse))
            for ci, test in enumerate(np.array(partial_mse) < threshold):
                logging.info(f"{ci}, {test}")
                if not test:
                    logging.info(f"cluster {ci} mse is not enough")
                    logging.info(points[ci])
                    logging.info(d_points[ci])
                    logging.info(jacobians[ci])



    def train(self, X, y, max_points=10, threshold=0.1, deep=False):
        min_mse = float('inf')
        target_partial_mse = []
        target_points = []
        target_d_points = []
        target_jacobians = []
        for n_point in range(1, max_points + 1):
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
    logging.basicConfig(filename='normal_JMLM.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %p %I:%M:%S')
    logging.info(f"===============Execution: JMLM===============")
    dataset = "mnist"
    X_train, X_test, y_train, y_test = load_data(dataset)
    logging.info(f"Dataset: {dataset}")

    start_time = time.time()
    jmlm = JMLM()
    # jmlm.deep_train(X_train, y_train, max_points=10, threshold=0.01)
    jmlm.train(X_train, y_train, max_points=50, threshold=0.01)

    y_pred = jmlm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f'Prediction Accuracy: {acc}')
    logging.info("--- %s seconds ---" % (time.time() - start_time))
    
    