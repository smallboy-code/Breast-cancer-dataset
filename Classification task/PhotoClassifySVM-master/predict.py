import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.svm import SVC
from get_eigenvalue import *


def multi_models_roc(names, sampling_methods, colors, list,X_test, y_test, save=True, dpin=100):
    """
    将多个机器模型的roc图输出到一张图上

    Args:
        names: list, 多个模型的名称
        sampling_methods: list, 多个模型的实例化对象
        save: 选择是否将结果保存（默认为png格式）

    Returns:
        返回图片对象plt
    """
    plt.figure(figsize=(20, 20), dpi=dpin)

    for (name, method, colorname,listname) in zip(names, sampling_methods, colors,list):
        # method.eval()

        y_test_preds = method.predict(listname)
        y_test_predprob = method.predict_proba(listname)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_test_predprob, pos_label=1)

        plt.plot(fpr, tpr, lw=5, label='{} (AUC={:.4f})'.format(name, auc(fpr, tpr)), color=colorname)
        plt.plot([0, 1], [0, 1], '--', lw=5, color='grey')
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xticks(np.arange(0, 1.1, step=0.1), fontsize=30)
        plt.yticks(np.arange(0.1, 1.1, step=0.1), fontsize=30)
        plt.xlabel('False Positive Rate', fontsize=30, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=30, fontweight='bold')
        plt.title('ROC Curve', fontsize=30, fontweight='bold')
        plt.legend(loc='lower right', fontsize=30)

    if save:
        plt.savefig('multi_models_roc.png')

    return plt

if __name__ == '__main__':
    pic = get_eigenvalue()
    pic.get_eigen()
    x_train, x_test, y_train, y_test = pic.get_data()
    clf_lbp = joblib.load('svm_LBP.m')
    clf_lpq = joblib.load('svm_LPQ.m')
    clf_glcm = joblib.load('svm_GLCM.m')
    x_test1 = pd.read_csv('pca_data/x_test_LBP.csv', index_col=0)
    list1 = x_test1.values.tolist()
    x_test2 = pd.read_csv('pca_data/x_test_LPQ.csv', index_col=0)
    list2 = x_test2.values.tolist()
    x_test3 = pd.read_csv('pca_data/x_test_GLCM.csv', index_col=0)
    list3 = x_test3.values.tolist()
    names = ['LBP-SVM',
             'LPQ-SVM',
             'GLCM-SVM']

    sampling_methods = [clf_lbp,
                        clf_lpq,
                        clf_glcm
                        ]

    colors = ['crimson',
              'steelblue',
              'gold'
              ]
    lists = [list1,list2,list3]
    # ROC curves
    train_roc_graph = multi_models_roc(names, sampling_methods, colors,lists, x_test, y_test, save=False)
    train_roc_graph.savefig('ROC_Train_all.png')


