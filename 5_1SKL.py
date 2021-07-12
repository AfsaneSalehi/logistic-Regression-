from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import exp
import scipy.stats as stats


def calc_measures(x, actual_classes, parameters):
    # Calculate all measures after prediction
	probab_thr=0.5
    predicteds = (predict(parameters, x) >= probab_threshold).astype(int)
    predicteds = predicteds.flatten()
    accuracy = np.mean(predicteds == actual_classes)
    accuracy = accuracy * 100
    tp = len(np.where((predicteds == actual_classes) & (predicteds == 1))[0])
	fp = len(np.where((predicteds != actual_classes) & (predicteds == 1))[0])
    tn = len(np.where((predicteds == actual_classes) & (predicteds == 0))[0])
    fn = len(np.where((predicteds != actual_classes) & (predicteds == 0))[0])
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f_score = (2*precision*recall)/(precision + recall)
    return accuracy, precision, recall, f_score


def cvalidation(data, folds):
    # K-fold cross validation
    bound = int(len(data)/folds)
    etas = [0.1, 0.3, 0.5, 0.7, 0.9]
    accuracies = []
    precisions = []
    recalls = []
    f_scores = []
    iters = []
    for eta in etas:
        max_iter = 0
        for i in range(folds):
            data_tmp = data.copy()
            test_set = data.iloc[i*bound:(i+1)*bound, :]
            train_set = data_tmp.drop(test_set.index)

            X_train = train_set.iloc[:, :-1]
            y_train = train_set.iloc[:, -1]
            X_train = np.c_[X_train]
            y_train = y_train[:, np.newaxis]

            X_test = train_set.iloc[:, :-1]
            y_test = train_set.iloc[:, -1]
            X_test = np.c_[X_test]
            y_test = y_test[:, np.newaxis]
            model = LogisticRegression( solver= 'lbfgs')
            model.fit(X_train, y_train.flatten())
            predicted_classes = model.predict(X_test)
            parameters = model.coef_
            iter = model.n_iter_
            if iter > max_iter:
                max_iter = iter
            acc, precision, recall, f_score  = calc_measures(X_test, y_test.flatten(), parameters, model)
            accuracies.append(acc)
            precisions.append(precision)
            recalls.append(recall)
            f_scores.append(f_score)

        iters.append(max_iter)
        print('\n', eta, '\n')
        print('\n Accuracy: ', np.mean(accuracies))
        print('\n Precision: ', np.mean(precisions))
        print('\n Recalls: ', np.mean(recalls))
        print('\n F_score: ', np.mean(f_scores))
        print('\n iters: ', max_iter)


if __name__== "__main__":
    data = pd.read_csv(r'D:\learning\University\AI_DrHarati\HWs\HW5\data_banknote_authentication.txt', header = None)
    data = data.sample(frac=1).reset_index(drop=True)
	folds = 5
    cvalidation(data, 5)

