import numpy as np
from sklearn.metrics import homogeneity_score, accuracy_score
from sklearn.cluster import KMeans

from load_data import load_data
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class KmeansClassifier:
    def __init__(self, n_clusters) -> None:
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.n_clusters = n_clusters

    def fit(self, X, y):
        if len(y) > 1:
            y = y.argmax(axis=1)
        
        fit_result = self.kmeans.fit(X)
        self.cluster_labels = self.infer_cluster_labels(self.kmeans, y)
        return self
    
    def acc(self, X, y):
        if len(y) > 1:
            y = y.argmax(axis=1)
        X_clusters = self.kmeans.predict(X)
        y_pred = self.infer_data_labels(X_clusters, self.cluster_labels)
        acc = accuracy_score(y, y_pred)
        return acc

    def infer_cluster_labels(self, kmeans, actual_labels):
        inferred_labels = {}
        for i in range(self.kmeans.n_clusters):
            labels = []
            index = np.where(self.kmeans.labels_ == i)
            labels.append(actual_labels[index])
            if len(labels[0]) == 1:
                counts = np.bincount(labels[0])
            else:
                counts = np.bincount(np.squeeze(labels))
            if counts.size:
                if np.argmax(counts) in inferred_labels:
                    inferred_labels[np.argmax(counts)].append(i)
                else:
                    inferred_labels[np.argmax(counts)] = [i]
        return inferred_labels  

    def infer_data_labels(self, X_labels, cluster_labels):
        predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)
        
        for i, cluster in enumerate(X_labels):
            for key, value in cluster_labels.items():
                if cluster in value:
                    predicted_labels[i] = key
                    
        return predicted_labels

    def calc_metrics(self, estimator, data, labels):
        print('Number of Clusters: {}'.format(estimator.n_clusters))
        inertia = estimator.inertia_
        print("Inertia: {}".format(inertia))
        homogeneity = homogeneity_score(labels, estimator.labels_)
        print("Homogeneity score: {}".format(homogeneity))
        return inertia, homogeneity


def low_dim(X_train, X_test, y_train, n_pca=0, pca=True, lda=True):
    if pca:
        if n_pca == 0:
            n_pca = int(X_train.shape[1]*0.8)
        pca = PCA(n_components=n_pca)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    if lda:
        clf = LinearDiscriminantAnalysis()
        X_train = clf.fit_transform(X_train, y_train.argmax(axis=1))
        X_test = clf.transform(X_test)
    return X_train, X_test

dataset = "mnist"
X_train, X_test, y_train, y_test = load_data(dataset, normalization=False)

def main():
    
    global X_train, X_test, y_train, y_test
    X_train, X_test = X_train / 255., X_test / 255.
    X_train, X_test = low_dim(X_train, X_test, y_train)

    KC = KmeansClassifier(n_clusters=300).fit(X_train, y_train)
    acc_train = KC.acc(X_train, y_train)
    acc_test = KC.acc(X_test, y_test)

    # print(KC.kmeans.cluster_centers_)
    


    print(f'Train Prediction Accuracy: {acc_train}')
    # print(f'Test Prediction Accuracy: {acc_test}')

if __name__ == '__main__':

    main()
