# %%

from scipy.sparse import diags # imports ##
from itertools import product
import numpy as np
import argparse
import uuid
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy import stats
import pickle
#import matplotlib.pyplot as plt
from scipy.sparse.linalg import inv
from scipy.linalg import pinv, svd
from scipy.sparse.linalg import svds

import gc
from sklearn.utils import shuffle
from scipy.sparse import diags
from scipy.sparse import csr_matrix
from scipy import linalg

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy import stats

# %%
def parse_arguments(parser):
    """Read user arguments"""
    parser.add_argument('--days', type=int, default=90,
                        help='Number of days before delivery')
    args = parser.parse_args()
    return args

unique_code = str(uuid.uuid4())[:8]


# PARSER = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# ARGS = parse_arguments(PARSER)

# days = [30, 90, 180, 270]
days = [0, 12, 24]
# days = [30]

    ##### Read the Dataframe #### 
# X = pd.read_csv('Dataframes/fullDataframe_30Days_full_code_int.csv')
# X.drop('Unnamed: 0', axis=1, inplace=True)
# y = X['Y']
# X.drop('Y', axis=1, inplace=True)

# est = sm.OLS(y, X)
# est2 = est.fit()
# print(est2.summary())

window = -1

def expand_grid(dictionary):
   return pd.DataFrame([row for row in product(*dictionary.values())], 
                       columns=dictionary.keys())

alphas = [0.001, 0.01, 0.1, 1]
penalties = ['l1', 'l2']

hyperp = {'alpha': alphas, 'penalty':penalties}
df_hyper = expand_grid(hyperp)

   
# %%
for day in days:
    #
    X = pd.read_csv('full_pre_ICD10/fullDataframe_'+str(day)+'_' + str(window) + 'Weeks_full_code.csv.gz')
    # X = pd.read_hdf('Dataframes/fullDataframe_'+str(day)+'Days_full_code_int.hd5')
    #    
    X.drop('Unnamed: 0', axis=1, inplace=True)
    #
    # original_df = shuffle(original_df)
    # print('Im here')
    #
    #original_df = original_df.sample(frac=0.5)
    #
    ## Flip zeros and ones of target variable
    #
    # df_one = original_df[original_df['Y']==1] #full term
    # df_zero = original_df[original_df['Y']==0] #preterm
    #
    #df_one = df_one.sample(frac=0.5)
    #
    #
    # df_one['Y'] = 0 #full term
    # df_zero['Y'] = 1 #pretrm 
    #
    #df_zero = df_zero.sample(frac=0.8)
    #
    # frames = [df_one, df_zero]
    #
    #
    # df = pd.concat(frames)
    #
    #
    # df=df.reset_index(drop=True)
    #
    #
    ## Uncomment to get separete dataframes for diagnosis, medications and procedures. ##
    #allColumns = df.columns
    #medication_columns = list()
    #diagnosis_columns = list()
    #procedure_columns = list()
    ##
    ##
    #for c in allColumns: 
    #    if c.startswith('M'):
    #        medication_columns.append(c)
    #    if c.startswith('D'):
    #        diagnosis_columns.append(c)
    #    if c.startswith('P'):
    #        procedure_columns.append(c)
    #
    #
    #medication_df = df[medication_columns]
    #medication_df['Y'] = df['Y']
    #
    #diagnosis_df = df[diagnosis_columns]
    #diagnosis_df['Y'] = df['Y']
    #
    #procedure_df = df[procedure_columns]
    #procedure_df['Y'] = df['Y']
    #
    #df = procedure_df
    #
    ## select one of the dataframes ##
    y = X['Y']
    X.drop('Y', axis=1, inplace=True)
    #
    fullTerm = X[y == 0]
    fullTermTotal = fullTerm.sum()
    fulltermTotaldf = pd.Series.to_frame(fullTermTotal)
    fulltermTotaldf.columns = ['totals-full-term']
    countFilter_df = fulltermTotaldf[fulltermTotaldf['totals-full-term']>10]
    #
    preterm = X[y == 1]
    pretermTotal = preterm.sum()
    pretermTotaldf = pd.Series.to_frame(pretermTotal)
    pretermTotaldf.columns = ['totals-preterm']
    countFilter_df2 = pretermTotaldf[pretermTotaldf['totals-preterm']>10]
    #
    A=countFilter_df.index
    B=X.columns
    #
    lst3 = [value for value in A if value in B] 
    #
    C = countFilter_df2.index
    #
    lst4 = [value for value in lst3 if value in C] 
    #
    #
    X = X[lst4]
    feat_names = X.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33) #, random_state=42)
    def batch(iterable_X, iterable_y, n=1):
        l = len(iterable_X)
        for ndx in range(0, l, n):
            yield iterable_X[ndx:min(ndx + n, l)], iterable_y[ndx:min(ndx + n, l)]
    #
    for k in range(df_hyper.shape[0]):
        alph = df_hyper['alpha'][k]
        pena = df_hyper['penalty'][k]
        logreg = linear_model.SGDClassifier(alpha=alph, loss='log', penalty=pena, n_jobs=-1, shuffle=True, max_iter=1000, verbose=0, tol=0.001)
        ROUNDS =10 
        for i in range(ROUNDS):
            batcherator = batch(X_train, y_train, 50000)
            print("Rounds {}".format(i))
            for index, (chunk_X, chunk_y) in enumerate(batcherator):
                logreg.partial_fit(chunk_X, chunk_y, classes=[0, 1])
        #
        #
        params = np.append(logreg.intercept_,logreg.coef_)
        #
        predictions =logreg.predict_proba(X_test)[:, 1]
        #
        predictions.shape
        fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
        pre, rec, thresholds = metrics.precision_recall_curve(y_test, predictions)
        #
        auc_roc = metrics.auc(fpr, tpr)
        # auc_pr = metrics.auc(rec, pre)   
        # print('AUC-ROC = {} , AUC-PRC = {}'.format(auc_roc, auc_pr))
        auc_pr = metrics.average_precision_score(y_test, predictions)
        #  print('AUC-ROC = {} , AUC-PRC = {}'.format(auc_roc, auc_pr))
        result = pd.DataFrame({'y_test':y_test, 'predictions': predictions})
        fname_res = 'results/predtestset_' + str(day) + '_' + 'alpha_' + str(alph) + 'penal_' + str(pena) + '_' + unique_code + '.pkl'
        result.to_pickle(fname_res)

