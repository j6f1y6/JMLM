import time
from jmlm import JMLM
from load_data import load_data
from sklearn.metrics import accuracy_score
if __name__ == '__main__':
    dataset = "mnist"
    X_train, X_test, y_train, y_test = load_data(dataset)
    start_time = time.time()


    jmlm = JMLM()
    n_max_center = 20
    jmlm.jjlm_train(X_train, y_train, n_max_center)
    
    y_pred = jmlm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('Prediction Accuracy after JMLM:', acc)
    print("--- %s seconds ---" % (time.time() - start_time))