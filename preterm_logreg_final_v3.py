from scipy.sparse import diags # imports ##
import json
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
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



days = [0, 12, 24]
windows = [-1, 0]

def batch(iterable_X, iterable_y, n=1):
    l = len(iterable_X)
    for ndx in range(0, l, n):
        yield iterable_X[ndx:min(ndx + n, l)], iterable_y[ndx:min(ndx + n, l)]


with open('hf_fp_dictionary_mod7_v3.6.json') as json_file:
    desc_dic = json.load(json_file)


for window in windows:
    for day in days:
        #
        if (window == 0) and (day == 0):
            continue
        X = pd.read_csv('full_pre_ICD10/fullDataframe_'+str(day)+'_' + str(window) + 'Weeks_full_code.csv.gz')
        #    
        X.drop('Unnamed: 0', axis=1, inplace=True)
        ## select one of the dataframes ##
        y = X['Y']
        X.drop('Y', axis=1, inplace=True)
        ######## Counts of Features ##########
        #    
        # 
        fullTerm = X[y == 0]
        fullTermTotal = fullTerm.sum()
        fulltermTotaldf = pd.Series.to_frame(fullTermTotal)
        fulltermTotaldf.columns = ['totals-full-term']
        countFilter_df = fulltermTotaldf[fulltermTotaldf['totals-full-term']>100]
        #
        preterm = X[y == 1]
        pretermTotal = preterm.sum()
        pretermTotaldf = pd.Series.to_frame(pretermTotal)
        pretermTotaldf.columns = ['totals-preterm']
        countFilter_df2 = pretermTotaldf[pretermTotaldf['totals-preterm']>100]
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
       
        ##### Logistic Regression
        # gc.collect()
        logreg = linear_model.SGDClassifier(alpha=.0001, loss='log', penalty='l2', n_jobs=-1, shuffle=True, max_iter=200, verbose=0, tol=0.001)
        ROUNDS = 30
        for i in range(ROUNDS):
            batcherator = batch(X, y, 5000)
            print("Rounds {}".format(i))
            for index, (chunk_X, chunk_y) in enumerate(batcherator):
                logreg.partial_fit(chunk_X, chunk_y, classes=[0, 1])
        #
        #
        #
        #
        params = np.append(logreg.intercept_,logreg.coef_)
        #
        predictions =logreg.predict_proba(X)[:, 1]
        #
        predictions.shape
        #
        #
        newX = csr_matrix(pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X)))
        #
        pred = predictions*(1-predictions)
        #
        # V = np.diagflat(pred)
        V = diags(pred)
        #    
        c = newX.T.dot(V).dot(newX).todense()
        # covariance = np.linalg.pinv(np.dot(np.dot(newX.T, V), newX)) 
        # u, s, vt = svds(c, k=c.shape[0])
        # u, s, vt = svd(c)
        covariance = pinv(c)
        # covariance = pinv(c.todense()) 
        se=np.sqrt(covariance.diagonal())              
        z_val = params/ se                             
        p_values2 = stats.norm.sf(abs(z_val))*2 
        pvalue_adj = multipletests(p_values2, alpha=0.05, method='bonferroni')
        #
        #
        #
        df2 = X.columns.to_list()
        cols = ["Constant"] + df2
        #
        #
        #### Format output file (highest p-values)
        #
        #
        myDF3 = pd.DataFrame()
        #
        #
        myDF3['names'],myDF3["Coefficients"],myDF3["pvalues2"], myDF3["pvalue_adj"], myDF3["se"]= [cols,params,p_values2, pvalue_adj[1], se]
        #
        myDF3 = myDF3.set_index(['names'])
        #
        myDF3= myDF3.drop(['Constant'])
        #myDF3= myDF3.drop(['Unnamed: 0'])
        myDF3.columns.name = myDF3.index.name
        myDF3.index.name = None
        #
        #
        pretermTotaldf.index.name = 'names'
        #
        #
        pretermTotaldf.columns.name = pretermTotaldf.index.name
        pretermTotaldf.index.name = None
        #
        #
        fulltermTotaldf.index.name = 'names'
        #
        #
        fulltermTotaldf.columns.name = fulltermTotaldf.index.name
        fulltermTotaldf.index.name = None
        #
        #
        frames = [myDF3, fulltermTotaldf, pretermTotaldf]
        #
        resultsDf = pd.concat(frames, axis=1)
        #
        #
        resultsDf = resultsDf.sort_values(by='pvalues2', ascending=False)
        #
        #
        clean_df = resultsDf[resultsDf['pvalue_adj'] <= 0.05]
        #
        #
        # f = open("diag_proc_med_lab_dicts.pkl", "rb")
       #
        descriptions = list()
        #
        clean_df['name'] = clean_df.index
        #
        for idx, row in clean_df.iterrows():
            if  clean_df.loc[idx,'name'].startswith('M'):
                clean_df.loc[idx,'name'] = clean_df.loc[idx,'name'] +'00000'
        #        
        #
        clean_df['description'] = clean_df['name'].map(desc_dic)
        #
        clean_df = clean_df.loc[:, clean_df.columns != 'name']
        #
        #cli_d['P_8354000000'] 
        #
        clean_df['odds_ratio'] = np.exp(clean_df['Coefficients'])
        clean_df['odds_ratio_95low'] = clean_df['odds_ratio'] - 1.96 * se
        clean_df['odds_ratio_95high'] = clean_df['odds_ratio'] + 1.96 * se
        clean_df['|Coefficients|'] = np.abs(clean_df['Coefficients'])
        #
        #
        #
        clean_df.to_csv('results/results_' + str(day) + 'days_' + str(window) + 'fullcode_optimizedLR_featsel.csv')



