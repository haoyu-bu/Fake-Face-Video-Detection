from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle
from sklearn import svm
import sklearn
import numpy as np

def classify(X, Y, f_X):
    '''
    train and test the SVM classifier.

    Args:
        X: features.
        Y: labels.
        f_X: fake features.

    Returns:
        A dict mapping evaluation metrics to the corresponding results.
    '''

    X = np.reshape(X, (np.shape(X)[0], -1))
    f_X = np.reshape(f_X, (np.shape(f_X)[0], -1))
    rus = RandomUnderSampler(return_indices=True)
    X, Y, id_rus = rus.fit_sample(X, Y)

    X, Y = shuffle(X, Y)
    X = np.array(X)
    f_X = shuffle(f_X)

    clf = svm.SVC(gamma='scale')
    result = {}

    skf = sklearn.model_selection.StratifiedKFold(n_splits=len(Y)/2)
    skf.get_n_splits(X, Y)
    fps = []
    tps = []
    i = 0
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # use fake videos to test
        neg = list(y_test).index(0)
        if i < len(f_X):
            X_test[neg] = f_X[i]
            i += 1
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        _, fp, _, tp = sklearn.metrics.confusion_matrix(y_test, predict).ravel()
        fps.append(fp)
        tps.append(tp)
    result['tp'] = np.mean(tps)
    result['fp'] = np.mean(fps)
    print result['tp'], result['fp']

    return result