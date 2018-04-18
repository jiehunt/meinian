
# coding: utf-8

# In[1]:


import pandas as pd
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import numpy as np


# In[2]:


get_ipython().run_cell_magic('time', '', 'data=pd.read_csv(\'cleaned_data_0418.csv\',low_memory=False)\ndata = data.rename(columns={"收缩压": "Systolic", "舒张压": "Diastolic", "血清甘油三酯":"triglyceride", "血清高密度脂蛋白":"HDL", "血清低密度脂蛋白":"LDL"})\ntest_lenth=9538\ntest=data[-test_lenth:]\ntrain=data[:-test_lenth]\ntest_vid = test[\'vid\']\ntarget=train.columns[1:6]\nprint (train.columns)\nprint (test.columns)\ncols = list(set(train.columns)- set(target))')


# In[3]:


for class_name in target:
    train = train[np.isfinite(train[class_name])]
    train = train[train[class_name] > 0]
    train = train[train[class_name] < 300]
y_train = train[target]

x_train=train[cols].select_dtypes(include=['float64'])
x_test=test[cols].select_dtypes(include=['float64'])

print ("OK")
X_train, X_valid, y_train, y_valid = train_test_split( x_train, y_train, test_size=0.1, random_state=42)

train_all = lgb.Dataset(x_train.values)
pred = pd.DataFrame()
pred_test = pd.DataFrame()
scores = []


# In[4]:


def my_error(pred, train_data):
    labels = train_data.get_label()
    res = np.power(np.log(pred + 1) - np.log(labels + 1) , 2)
    return 'error', np.mean(res) , False


# In[5]:



lgb_param_s = {
        'task' : 'train', 'boosting_type' : 'gbdt', 'objective' : 'regression',
        'metric' : {'l2'},
        'num_leaves' : 31*2, 'learning_rate' : 0.05, 'feature_fraction' : 0.9,
        'bagging_fraction' : 0.8, 'bagging_freq': 5, 'verbose': 1
       # 'scale_pos_weight':40., # because training data is extremely unbalanced
}
lgb_param_d = {
        'task' : 'train', 'boosting_type' : 'gbdt', 'objective' : 'regression',
        'metric' : {'l2'},
        'num_leaves' : 31, 'learning_rate' : 0.05, 'feature_fraction' : 0.9,
        'bagging_fraction' : 0.8, 'bagging_freq': 5, 'verbose': 1
       # 'scale_pos_weight':40., # because training data is extremely unbalanced
}
lgb_param_t = {
        'task' : 'train', 'boosting_type' : 'gbdt', 'objective' : 'regression',
        'metric' : {'l2'},
        'num_leaves' : 31, 'learning_rate' :0.05, 'feature_fraction' : 0.9,
        'bagging_fraction' : 0.8, 'bagging_freq': 5, 'verbose': 1
       # 'scale_pos_weight':40., # because training data is extremely unbalanced
}
lgb_param_h = {
        'task' : 'train', 'boosting_type' : 'gbdt', 'objective' : 'regression',
        'metric' : {'l2'},
        'num_leaves' : 31, 'learning_rate' : 0.05, 'feature_fraction' : 0.9,
        'bagging_fraction' : 0.8, 'bagging_freq': 5, 'verbose': 1
       # 'scale_pos_weight':40., # because training data is extremely unbalanced
}
lgb_param_l = {
        'task' : 'train', 'boosting_type' : 'gbdt', 'objective' : 'regression',
        'metric' : {'l2'},
        'num_leaves' : 31, 'learning_rate' :0.05, 'feature_fraction' : 0.9,
        'bagging_fraction' : 0.8, 'bagging_freq': 5, 'verbose': 1
       # 'scale_pos_weight':40., # because training data is extremely unbalanced
}
lgb_param_set = { 'Systolic': lgb_param_s ,
              'Diastolic': lgb_param_s ,
              'triglyceride': lgb_param_s,
              'HDL': lgb_param_s,
              'LDL': lgb_param_s
              }
for class_name in target:
    dtrain = lgb.Dataset(X_train, label=y_train[class_name].values)
    dvalid = lgb.Dataset(X_valid, label=y_valid[class_name].values)
    evals_results = {}
    print ("goto train", class_name)
    bst = lgb.train(lgb_param_set[class_name],
                    dtrain,
                    valid_sets=dvalid,
                    evals_result=evals_results,
                    num_boost_round=1000,
                    early_stopping_rounds=50,
                    verbose_eval=20,
                    feval=my_error,
               )

    pred['pred_'+str(class_name)] = bst.predict(x_train, num_iteration=bst.best_iteration)
    pred_test[str(class_name)] = bst.predict(x_test, num_iteration=bst.best_iteration)
    scores.append(bst.best_score['valid_0']['l2'])
    print (bst.best_score['valid_0']['l2'])


# In[ ]:


res = pd.DataFrame()
res = pd.concat([y_train,pred], axis=1 )
res.to_csv("res.csv", index=False)

for i in range(0, len(scores)):
    print (scores[i])

print (np.mean(scores))

sub = pd.concat([test_vid, pred_test], axis=1)
sub.to_csv("submission_379.csv", index=False, header=False)

