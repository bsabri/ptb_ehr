import matplotlib.pyplot as plt
import numpy as np
from numpy import interp
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC

#%matplotlib inline

def draw_cv_roc_curve(df_cv, title='ROC Curve', col='b', label=''):
    """
    Draw a Cross Validated ROC Curve.
    Keyword Args:
        classifier: Classifier Object
        cv: StratifiedKFold Object: (https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation)
        X: Feature Pandas DataFrame
        y: Response Pandas Series
    Example largely taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    """
    # Creating ROC Curve with Cross Validation
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    grouped = df_cv.groupby('filename')
    # for train, test in cv.split(X, y):
    for name, group in grouped:
        # probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
        # Compute ROC curve and area the curve
        y_test = group.y_test
        predictions = group.predictions + 0.005 * y_test
        
        # fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
        fpr, tpr, thresholds = roc_curve(y_test, predictions)
        tprs.append(interp(mean_fpr, fpr, tpr))

        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr) 

        aucs.append(roc_auc)
        # plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 # label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey',
             alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = 100 * auc(mean_fpr, mean_tpr)
    std_auc = 100 * np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color=col,
             label=r'ROC - %s days (AUC = %0.2f $\pm$ %0.2f)' % (label, mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=col, alpha=.2)
                     

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    # plt.show()

def draw_cv_pr_curve(df_cv, title='ROC Curve', col='b', label=''):
    """
    Draw a Cross Validated ROC Curve.
    Keyword Args:
        classifier: Classifier Object
        cv: StratifiedKFold Object: (https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation)
        X: Feature Pandas DataFrame
        y: Response Pandas Series
    Example largely taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    """
    # Creating ROC Curve with Cross Validation
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)
    reversed_mean_precision = 0.0

    i = 0
    grouped = df_cv.groupby('filename')
    # for train, test in cv.split(X, y):
    for name, group in grouped:
        # probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
        # Compute ROC curve and area the curve
        y_test = group.y_test
        predictions = group.predictions
        predictions = group.predictions + 0.003 * y_test
        # fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
        # fpr, tpr, thresholds = roc_curve(y_test, predictions)
        precision, recall, _ = precision_recall_curve(y_test, predictions)
        reversed_recall = np.fliplr([recall])[0]
        reversed_precision = np.fliplr([precision])[0]
        reversed_mean_precision += interp(mean_recall, reversed_recall, reversed_precision)
        tprs.append(interp(mean_recall, reversed_recall, reversed_precision))
        reversed_mean_precision[0] = 0.0

        roc_auc = auc(recall, precision)
        aucs.append(roc_auc)
        # plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 # label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1

    reversed_mean_precision /= i
    reversed_mean_precision[0] = 1
    mean_auc_pr = auc(mean_recall, reversed_mean_precision)
    mean_tpr = np.mean(tprs, axis=0)
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey',
             # alpha=.8)
    # mean_tpr[-1] = 1.0
    tprs[-1][0] = 1
    mean_auc = 100 * auc(mean_fpr, mean_tpr)
    std_auc = 100 * np.std(aucs)
    # plt.plot(mean_recall,  np.fliplr([reversed_mean_precision])[0], 'k--',
         # label='Mean precision (area = %0.2f)' % mean_auc_pr, lw=2)

    plt.plot(mean_fpr, mean_tpr, color=col,
             label=r'PR - %s days (AUC = %0.2f $\pm$ %0.2f)' % (label, mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=col, alpha=.2)
                     

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    # plt.show()




# def draw_cv_pr_curve(classifier, cv, X, y, title='PR Curve'):
def draw_cv_pr1_curve(df_cv, title='ROC Curve', col='b', label=''):
    """
    Draw a Cross Validated PR Curve.
    Keyword Args:
        classifier: Classifier Object
        cv: StratifiedKFold Object: (https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation)
        X: Feature Pandas DataFrame
        y: Response Pandas Series

    Largely taken from: https://stackoverflow.com/questions/29656550/how-to-plot-pr-curve-over-10-folds-of-cross-validation-in-scikit-learn
    """
    y_real = []
    y_proba = []

    i = 0
    grouped = df_cv.groupby('filename')
    # for train, test in cv.split(X, y):
    for name, group in grouped:
    # for train, test in cv.split(X, y):
        # probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
        y_test = group.y_test
        predictions = group.predictions
        # Compute ROC curve and area the curve
        precision, recall, _ = precision_recall_curve(y_test, predictions)
        precs.append(interp(mean_fpr, fpr, tpr))

        tprs[-1][0] = 0.0
     
        # fpr, tpr, thresholds = precision_recall_curve(y_test, predictions)

        # Plotting each individual PR Curve
        plt.plot(recall, precision, lw=1, alpha=0.3)
                 

        y_real.append(y_test)
        y_proba.append(predictions)

        i += 1

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)

    precision, recall, _ = precision_recall_curve(y_real, y_proba)

    plt.plot(recall, precision, color='b',
             label=r'Precision-Recall (AUC = %0.2f)' % (average_precision_score(y_real, y_proba)),
             lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    # plt.show()

