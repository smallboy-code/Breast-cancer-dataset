from time import time

import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC

# load data
#senet features
# x_train_se = np.loadtxt('./features/x_train_se.txt',delimiter=',')
# y_train_se = np.loadtxt('./features/y_train_se.txt',delimiter=',').flatten()
# x_test_se = np.loadtxt('./features/x_test_se.txt',delimiter=',')
# y_test_se = np.loadtxt('./features/y_test_se.txt',delimiter=',')

#vit features
# x_train_vit = np.loadtxt('./features/x_train_vit.txt',delimiter=',')
# y_train_vit = np.loadtxt('./features/y_train_vit.txt',delimiter=',').flatten()
# x_test_vit = np.loadtxt('./features/x_test_vit.txt',delimiter=',')
# y_test_vit = np.loadtxt('./features/y_test_vit.txt',delimiter=',')

x_train = np.loadtxt('./features/x_train_all.txt',delimiter=',')
y_train = np.loadtxt('./features/y_train_all.txt',delimiter=',').flatten()
x_test = np.loadtxt('./features/x_test_all.txt',delimiter=',')
y_test = np.loadtxt('./features/y_test_all.txt',delimiter=',')
#

t0 = time()
clf = SVC(C=0.2,kernel='rbf',probability=True)
clf = clf.fit(x_train,y_train)

joblib.dump(clf, 'svm2.m')
y_test_pred = clf.predict(x_test)
target_name = ['benign','malignant']
print(classification_report(y_test,y_test_pred,target_names=target_name,digits=4))
print("完成，共计 %0.3f秒" % (time() - t0))

#
# row = 0
# for line in lines:
#     line = line.strip().split('\t')
#     datamat[row, :] = line[::]
#     row += 1
#
# print(datamat)
# print(datamat.shape)
#
#
