import time
import logging

from jmlm import JMLM
from load_data import load_data
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

if __name__ == '__main__':
    logging.basicConfig(filename='normal_JMLM.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %p %I:%M:%S')
    logging.info(f"===============Execution: JJLM===============")

    dataset = "mnist"
    X_train, X_test, y_train, y_test = load_data(dataset)

    pca = PCA(n_components=300)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    clf = LinearDiscriminantAnalysis()
    X_train = clf.fit_transform(X_train, y_train.argmax(axis=1))
    X_test = clf.transform(X_test)

    start_time = time.time()


    jmlm = JMLM()
    n_max_center = 20
    jmlm.jjlm_train(X_train, y_train, n_max_center)
    
    y_pred = jmlm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f'Prediction Accuracy: {acc}')
    logging.info("--- %s seconds ---" % (time.time() - start_time))