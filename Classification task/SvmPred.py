import joblib
import numpy as np
from sklearn.metrics import classification_report, roc_curve, auc

x_test = np.loadtxt('./features/x_test_all.txt',delimiter=',')
y_test = np.loadtxt('./features/y_test_all.txt',delimiter=',')
path = 'svm2.m'
clf = joblib.load(path)
y_test_pred = clf.predict(x_test)
y_test_predprob = clf.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_predprob, pos_label=1)
target_name = ['benign','malignant']
print(classification_report(y_test,y_test_pred,target_names=target_name,digits=4))
print('auc: ',auc(fpr,tpr))
