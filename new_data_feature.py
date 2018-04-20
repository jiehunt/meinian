
# coding: utf-8

# In[14]:





import pandas as pd
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import numpy as np


# In[2]:
#数据集已经上传到git
data=pd.read_csv('float_clean_data.csv',low_memory=False)
data = data.rename(columns={"收缩压": "Systolic", "舒张压": "Diastolic", "血清甘油三酯":"triglyceride", "血清高密度脂蛋白":"HDL", "血清低密度脂蛋白":"LDL"})
test_lenth=9538
test=data[-test_lenth:]
train=data[:-test_lenth]
test_vid = test['vid']
target=["Systolic","Diastolic","triglyceride","HDL","LDL"]
print (train.columns)
print (test.columns)
cols = list(set(train.columns)- set(target))

# In[3]:



# In[15]:



a=train[target].describe(percentiles=[.005,.05,.10,.20,.25,.5,.75,.95,.99,.995])
for class_name in target:
    train = train[np.isfinite(train[class_name])]
    train = train[train[class_name] > a.iloc[4,:][class_name]]
    train = train[train[class_name] < a.iloc[-2,:][class_name]]
y_train = train[target]

x_train=train[cols].select_dtypes(include=['float64'])
x_test=test[cols].select_dtypes(include=['float64'])

print ("OK")
X_train, X_valid, y_train, y_valid = train_test_split( x_train, y_train, test_size=0.15, random_state=42)

train_all = lgb.Dataset(x_train.values)
pred = pd.DataFrame()
pred_test = pd.DataFrame()



# In[4]:


def my_error(pred, train_data):
    labels = train_data.get_label()
    res = np.power(np.log(pred + 1) - np.log(labels + 1) , 2)
    return 'error', np.mean(res) , False


# In[16]:




# In[5]:



# In[17]:


scores = []

lgb_param_s = {
        'task' : 'train', 'boosting_type' : 'gbdt', 'objective' : 'regression',
        'metric' : {'error'},
        'num_leaves' :100, 'learning_rate' : 0.01, 'feature_fraction' : 0.9,
        'bagging_fraction' : 0.8, 'bagging_freq': 5, 'verbose': 1
       # 'scale_pos_weight':40., # because training data is extremely unbalanced
}
lgb_param_d = {
        'task' : 'train', 'boosting_type' : 'gbdt', 'objective' : 'regression',
        'metric' : {'error'},
        'num_leaves' : 100, 'learning_rate' : 0.01, 'feature_fraction' : 0.8,
        'bagging_fraction' : 0.8, 'bagging_freq': 5, 'verbose': 1
       # 'scale_pos_weight':40., # because training data is extremely unbalanced
}
lgb_param_t = {
        'task' : 'train', 'boosting_type' : 'dart', 'objective' : 'regression',
        'metric' : {'error'},
        'num_leaves' : 62, 'learning_rate' :0.05, 'feature_fraction' : 0.9,
        'bagging_fraction' : 0.8, 'bagging_freq': 5, 'verbose': 1
       # 'scale_pos_weight':40., # because training data is extremely unbalanced
}
lgb_param_h = {
        'task' : 'train', 'boosting_type' : 'gbdt', 'objective' : 'regression',
        'metric' : {'error'},
        'num_leaves' : 100, 'learning_rate' : 0.3, 'feature_fraction' : 0.8,
        'bagging_fraction' : 0.8, 'bagging_freq': 5, 'verbose': 1
       # 'scale_pos_weight':40., # because training data is extremely unbalanced
}
lgb_param_l = {
        'task' : 'train', 'boosting_type' : 'gbdt', 'objective' : 'regression',
        'metric' : {'error'},
        'num_leaves' : 100, 'learning_rate' :0.01, 'feature_fraction' : 0.8,
        'bagging_fraction' : 0.8, 'bagging_freq': 5, 'verbose': 1
       # 'scale_pos_weight':40., # because training data is extremely unbalanced
}
lgb_param_set = { 'Systolic': lgb_param_s ,
              'Diastolic': lgb_param_d ,
              'triglyceride': lgb_param_t,
              'HDL': lgb_param_h,
              'LDL': lgb_param_l,
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
                    early_stopping_rounds=100,
                    verbose_eval=50,
                    feval=my_error,
               )

    pred['pred_'+str(class_name)] = bst.predict(x_train, num_iteration=bst.best_iteration)
    pred_test[str(class_name)] = bst.predict(x_test, num_iteration=bst.best_iteration)
    scores.append(bst.best_score['valid_0']['error'])
    print (bst.best_score['valid_0']['error'])


# class_name='triglyceride'
# dtrain = lgb.Dataset(X_train, label=y_train[class_name].values)
# dvalid = lgb.Dataset(X_valid, label=y_valid[class_name].values)
# evals_results = {}
# print ("goto train", class_name)
# bst = lgb.train(lgb_param_set[class_name],
#                 dtrain,
#                 valid_sets=dvalid,
#                 evals_result=evals_results,
#                 num_boost_round=1000,
#                 early_stopping_rounds=100,
#                 verbose_eval=50,
#                 feval=my_error,
#            )

# In[ ]:


# res = pd.DataFrame()
# res = pd.concat([y_train,pred], axis=1 )
# res.to_csv("res.csv", index=False)
print("******各组得分******* ")
for i in range(0, len(scores)):
    print (scores[i])
print("******平均得分******* ")
print (np.mean(scores))
# sub=pd.DataFrame()
# sub['vid']=test_vid.values
# sub = pd.concat([sub, pred_test], axis=1)
# sub.to_csv("submission_379.csv", index=False, header=False)


# In[17]:


sub=pd.DataFrame()
sub['vid']=test_vid.values
sub = pd.concat([sub, pred_test], axis=1)
sub.to_csv("submission_420.csv", index=False, header=False)


# In[ ]:


0.0173004166096
0.0193812210712
0.0893883024543
0.0123830033026
0.0352206543348
******平均得分******* 
0.0347347195545

