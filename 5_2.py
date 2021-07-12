from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

data = loadmat(r'D:\learning\University\AI_DrHarati\HWs\HW5\faces.mat')
data = data['faces'].T
data = np.reshape(data, (400, 4096))
train = data[:300]
test = data[300:]
X_t = []
y_t = []
X_t = []
y_t = []

for t in test:
    t = np.reshape(t, (64, 64)).T
    tx = t[:][0:32]
    ty = t[:][32:64]
    X_t.append(np.reshape(tx, (2048)))
    y_t.append(np.reshape(ty, (2048)))

X_t = np.array(X_t)
y_t = np.array(y_t)	

for t in train:
    t = np.reshape(t, (64, 64)).T
    tx = t[:][0:32]
    ty = t[:][32:64]
    X_t.append(np.reshape(tx, (2048)))
    y_t.append(np.reshape(ty, (2048)))
    
X_t = np.array(X_t)
y_t = np.array(y_t)

W = np.dot(X_t.T, X_t)
W = np.linalg.inv(W)
W = np.dot(W, X_t.T)
W = np.dot(W,y_t)

hs = np.dot(X_t, W)

rmse = np.sqrt(((hs - y_t) ** 2).mean())
print('RMSE= ', rmse)


