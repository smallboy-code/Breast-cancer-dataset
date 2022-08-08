import pickle

import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.svm import SVC
from get_eigenvalue import *
import numpy as np

pic = get_eigenvalue()
pic.get_eigen()
x_train, x_test, y_train, y_test = pic.get_data()
target_n = pic.get_target_n()
x_test1 = pd.read_csv('pca_data/x_test_T2WI.csv', index_col=0)
list = x_test1.values.tolist()
clf_lbp = joblib.load('svm_T2WI.m')
y_test_preds = clf_lbp.predict(list)
y_test_predprob = clf_lbp.predict_proba(list)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_predprob, pos_label=1)
a = auc(fpr,tpr)
print(classification_report(y_test,y_test_preds,target_names=target_n,digits=4))
print('auc: ',auc(fpr,tpr))
