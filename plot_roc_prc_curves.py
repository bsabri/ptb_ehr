# %%
import os

from importlib import reload
import draw_curves as dc
import matplotlib.pyplot as plt
reload(dc)

import glob
import pandas as pd

colors = ['b', 'g', 'r']
labels = ['0', '12', '24']


# labels = ['0', '12']


# %%
for i, l in enumerate(labels):
    files = glob.glob('results_old1/rf_predtestset_' + l + '_*.csv')
    # df = pd.concat([pd.read_csv(f, encoding='latin1').assign(filename=os.path.basename(f)) for f in files])
    # files = glob.glob('results_old1/predtestset_' + l + '_*.pkl')
    df = pd.concat([pd.read_pickle(f).assign(filename=os.path.basename(f)) for f in files])
    # dc.draw_cv_pr_curve(df, title='Precision-Recall Curves - Random Forests', col=colors[i], label= l)
    # dc.draw_cv_pr_curve(df, title='Precision-Recall Curves - Logistic Regression', col=colors[i], label= l)
    # dc.draw_cv_roc_curve(df, title='ROC Curves - Random Forests', col=colors[i], label= l)
    dc.draw_cv_roc_curve(df, title='ROC Curves - Logistic Regression', col=colors[i], label= l)

)

# plt.savefig('results/precision_recall_LR.pdf')
# plt.savefig('results/precision_recall_RF.pdf')
plt.savefig('results/roc_LR.pdf')
# plt.savefig('results/roc_RF.pdf')



# %%


