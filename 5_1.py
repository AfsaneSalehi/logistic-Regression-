import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from math import exp

	
def probability(theta, x):
    # Returns the probability after passing through sigmoid
	ins = np.dot(x, theta)
    return sigmoid(ins)

def cost_function(theta, x, y):
    m = x.shape[0]
    total_cost = -(1 / m) * np.sum(
        y * np.log(probability(theta, x)) + (1 - y) * np.log(
            1 - probability(theta, x)))
    return total_cost

def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))
	
def grad(theta, x, y):
    m = x.shape[0]
	ins = np.dot(x, theta)
	gr = (1 / m) * np.dot(x.T, sigmoid(ins) - y)
    return gr


def opt(eta, x, y):
    # Model fitting using gradient descent
    costs = []
    w = np.zeros((x.shape[1], 1))
    m = x.shape[0]
    d = -0.00000001381
    iters = 0
    while(iters < 2):
        gradient_vector = gradient(w, x, y)
        w -= eta * gradient_vector
        cost = cost_function(w, x, y)
        costs.append(cost)
        iters += 1

	d_n = costs[-1] - costs[-2]
    while d_n < d:
        gradient_vector = grad(w, x, y)
        w = w - eta * gradient_vector
        cost = cost_function(w, x, y)
        costs.append(cost)
		d_n = costs[-1] - costs[-2]
        iters += 1
		

    return w, iters


def predictions(parameters, x):
    # Predict classes
    parameters = parameters.T
    parameters = parameters[0]
    theta = parameters[:, np.newaxis]
    return probability(theta, x)


def calc_measures(x, actual_classes, parameters):
    # Calculate all measures after prediction
	probab_thr=0.5
    predicteds = (predictions(parameters, x) >= probab_threshold).astype(int)
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
	all = []
    for eta in etas:
        max_iter = 0
        for i in range(folds):
            data_tmp = data.copy()
            test_set = data.iloc[i*bound:(i+1)*bound, :]
            train_set = data_tmp.drop(test_set.index)
            X_train = train_set.iloc[:, :-1]
			X_train = np.c_[X_train]
			X_test = train_set.iloc[:, :-1]
            X_test = np.c_[X_test]           
			y_train = train_set.iloc[:, -1]
            y_train = y_train[:, np.newaxis]
            y_test = train_set.iloc[:, -1]
            y_test = y_test[:, np.newaxis]
            parameters, iter = opt(eta, X_train, y_train)
            if iter > max_iter:
                max_iter = iter
            acc, precision, recall, f_score  = calc_measures(X_test, y_test.flatten(), parameters)
            accuracies.append(acc)
            precisions.append(precision)
            recalls.append(recall)
            f_scores.append(f_score)

        iters.append(max_iter)
        print(eta, '\n')
        print('\n Accuracy: ', np.mean(accuracies))
        print('\n Precision: ', np.mean(precisions))
        print('\n Recalls: ', np.mean(recalls))
        print('\n F_score: ', np.mean(f_scores))
        print('\n iters: ', max_iter)
        all.append((np.mean(accuracies) + np.mean(precisions) + np.mean(recalls) + np.mean(f_scores)) / 4)
    plt.figure()
    plt.plot(etas, iters, lw=2, color='blue',
    label='LR / iterations')
    plt.figure()
    plt.plot(etas, all_means, lw=2, color='blue',
    label='LR / Mean of all measures')
    plt.show()


if __name__== "__main__":
    data = pd.read_csv(r'D:\learning\University\AI_DrHarati\HWs\HW5\data_banknote_authentication.txt', header = None)
    data = data.sample(frac=1).reset_index(drop=True)
	folds = 5
    cvalidation(data, folds)


