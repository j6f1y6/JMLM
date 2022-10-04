from jmlm import JMLM
from load_data import load_data
from sklearn.metrics import accuracy_score
if __name__ == '__main__':
    dataset = "mnist"
    X_train, X_test, y_train, y_test = load_data(dataset)

    jmlm = JMLM()
    n_iterations = 1
    n_max_center = 500
    jmlm.train(X_train, y_train, n_iterations, n_max_center)

    y_pred = jmlm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('Prediction Accuracy after JMLM:', acc)