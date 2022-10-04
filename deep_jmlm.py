from jmlm import JMLM
from load_data import load_data
from sklearn.metrics import accuracy_score
if __name__ == '__main__':
    dataset = "mnist"
    X_train, X_test, y_train, y_test = load_data(dataset)

    jmlm = JMLM()
    deep_iteration = 100
    max_kmeans_width = 5
    jmlm.deep_train(X_train, y_train, deep_iteration, max_kmeans_width)
    
    y_pred = jmlm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('Prediction Accuracy after JMLM:', acc)