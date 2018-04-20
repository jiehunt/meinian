import pandas as pd
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import  GridSearchCV
import xgboost as xgb
import time
from contextlib import contextmanager
from sklearn.metrics import make_scorer

@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def my_error(pred, train_data):
    labels = train_data.get_label()
    res = np.power(np.log(pred + 1) - np.log(labels + 1) , 2)
    return 'error', np.mean(res) , False

def xgb_error(pred, train_data):
    labels = train_data.get_label()
    res = np.power(np.log(pred + 1) - np.log(labels + 1) , 2)
    return 'error', np.mean(res)

def xgb_grd_error(pred, train_data):
    print (train_data.shape)
    labels = train_data
    res = np.power(np.log(pred + 1) - np.log(labels + 1) , 2)
    return np.mean(res)


xgb_loss  = make_scorer(xgb_grd_error, greater_is_better=False)
# lgb_loss  = make_scorer(my_error, greater_is_better=False)

# train=pd.read_csv('input/train_0415.csv',low_memory=False)
# test = pd.read_csv('input/test_0415.csv',low_memory=False)
train=pd.read_csv('input/train_0420-cnt.csv',low_memory=False)
test = pd.read_csv('input/test_0420-cnt.csv',low_memory=False)

test_vid = test['vid']
# temp = pd.read_csv('input/cleaned_data_0418.csv')
# temp = temp.rename(columns={"收缩压": "Systolic", "舒张压": "Diastolic", "血清甘油三酯":"triglyceride", "血清高密度脂蛋白":"HDL", "血清低密度脂蛋白":"LDL"})
# test = pd.read_csv('input/test_041922-no2.csv',low_memory=False)
# test_vid = test['vid']
# train=temp[:-len(test)]
# test = temp[-len(test):]

# train = train.rename(columns={"收缩压": "Systolic", "舒张压": "Diastolic", "血清甘油三酯":"triglyceride", "血清高密度脂蛋白":"HDL", "血清低密度脂蛋白":"LDL"})

target = [ 'Systolic', 'Diastolic', 'triglyceride', 'HDL', 'LDL' ]

cols = list(set(train.columns)- set(target))

train['triglyceride'] = train['triglyceride'].apply(lambda x: x if '>' not in str(x) else x[2:])
train['triglyceride'] = train['triglyceride'].apply(lambda x: float(x[:-1]) if str(x).endswith('.') else float(x))

# for class_name in target_out:
#     train = train[np.isfinite(train[class_name])]
#     train = train[train[class_name] > 0]
#     train = train[train[class_name] < target_out[class_name]]

a=train[target].describe(percentiles=[.005,.05,.10,.20,.25,.5,.75,.95,.99,.995])
for class_name in target:
    train = train[np.isfinite(train[class_name])]
    train = train[train[class_name] > a.iloc[4,:][class_name]]
    train = train[train[class_name] < a.iloc[-2,:][class_name]]

train.reset_index(drop=True, inplace=True)
print (train.shape)

y_train = train[target]
x_train=train[cols].select_dtypes(include=['float64','int64'])
print (x_train.shape)

nouse_list = []
for col in x_train.columns:
    if len(pd.unique(x_train[col])) == 1:
        nouse_list.append(col)

x_train=x_train.drop(nouse_list, axis=1)
print (x_train.shape)

test = test[x_train.columns]
nouse_test_list = []
for col in test.columns:
    if (test[[col]].dtypes[0]) == 'object':
        nouse_test_list.append(col)

cols = list (set(x_train.columns) - set(nouse_test_list))

nouse2_list=['0203','0209','0702','0703','0705','0706','0709','0726','0730','0731','3601',
             '1308','1316']

cols = list (set(cols) - set(nouse2_list))

x_train = x_train[cols]
test = test[cols]
print (x_train.shape)
print (test.shape)

print ("OK")

train_all = lgb.Dataset(x_train.values)
pred = pd.DataFrame()
pred_test = pd.DataFrame()
scores = []

X_train, X_valid, Y_train, Y_valid = train_test_split( x_train, y_train, test_size=0.15, random_state=42)
#################################
#  LGB Model
#################################
# lgb_param_s = {
#         'task' : 'train', 'boosting_type' : 'gbdt', 'objective' : 'regression',
#         'metric' : {'error'},
#         'lambda_l2':0.9,
#         'num_leaves' : 16, 'learning_rate' : 0.1, 'feature_fraction' : 0.8,
#         'bagging_fraction' : 0.9, 'bagging_freq': 5, 'verbose': 1
#        # 'scale_pos_weight':40., # because training data is extremely unbalanced
# }
# lgb_param_d = {
#         'task' : 'train', 'boosting_type' : 'gbdt', 'objective' : 'regression',
#         'metric' : {'error'},
#         'num_leaves' : 31, 'learning_rate' : 0.1, 'feature_fraction' : 0.9,
#         'bagging_fraction' : 0.8, 'bagging_freq': 5, 'verbose': 1
#        # 'scale_pos_weight':40., # because training data is extremely unbalanced
# }
# lgb_param_t = {
#         'task' : 'train', 'boosting_type' : 'gbdt', 'objective' : 'regression',
#         'metric' : {'l2'},
#         'num_leaves' : 31, 'learning_rate' : 0.1, 'feature_fraction' : 0.9,
#         'bagging_fraction' : 0.8, 'bagging_freq': 5, 'verbose': 1
#        # 'scale_pos_weight':40., # because training data is extremely unbalanced
# }
# lgb_param_h = {
#         'task' : 'train', 'boosting_type' : 'gbdt', 'objective' : 'regression',
#         'metric' : {'l2'},
#         'num_leaves' : 31, 'learning_rate' : 0.1, 'feature_fraction' : 0.9,
#         'bagging_fraction' : 0.8, 'bagging_freq': 5, 'verbose': 1
#        # 'scale_pos_weight':40., # because training data is extremely unbalanced
# }
# lgb_param_l = {
#         'task' : 'train', 'boosting_type' : 'gbdt', 'objective' : 'regression',
#         'metric' : {'l2'},
#         'num_leaves' : 31, 'learning_rate' : 0.1, 'feature_fraction' : 0.9,
#         'bagging_fraction' : 0.8, 'bagging_freq': 5, 'verbose': 1
#        # 'scale_pos_weight':40., # because training data is extremely unbalanced
# }
# lgb_param_set = { 'Systolic': lgb_param_s ,
#               'Diastolic': lgb_param_s ,
#               'triglyceride': lgb_param_s,
#               'HDL': lgb_param_s,
#               'LDL': lgb_param_s
#               }
# for class_name in target:
#     dtrain = lgb.Dataset(X_train, label=Y_train[class_name].values)
#     dvalid = lgb.Dataset(X_valid, label=Y_valid[class_name].values)
#     evals_results = {}
#     print ("goto train", class_name)
#     bst = lgb.train(lgb_param_set[class_name],
#                     dtrain,
#                     valid_sets=dvalid,
#                     evals_result=evals_results,
#                     num_boost_round=1000,
#                     early_stopping_rounds=50,
#                     verbose_eval=20,
#                     feval=my_error,
#                )
#
#     pred['pred_'+str(class_name)] = bst.predict(x_train, num_iteration=bst.best_iteration)
#     pred_test[str(class_name)] = bst.predict(test, num_iteration=bst.best_iteration)
#     scores.append(bst.best_score['valid_0']['error'])
#     print (bst.best_score['valid_0']['error'])
#     # scores.append(bst.best_score['valid_0']['l2'])
#     # print (bst.best_score['valid_0']['l2'])
#
# for i in range(0, len(scores)):
#     print (scores[i])
# print (np.mean(scores))
#
# res = pd.DataFrame()
# res = pd.concat([y_train,pred], axis=1 )
# res.to_csv("res.csv", index=False)

#################################
#  CV for train  xgb
#################################
K = 5
kf = KFold(n_splits = K, random_state = 1982, shuffle = True)
dtrain_all = xgb.DMatrix(x_train)

params = {
        'booster' : 'gbtree',
        'seed' : 1001,
        'silent' : 1,
        }
params['eta']=0.05
params['max_depth']=5
params['min_child_weight']=6
params['gamma']=0.2
params['subsample']=0.9
params['colsample_bytree']=0.7
params['colsample_bylevel']=1
params['reg_alpha']=10
params['reg_lambda']=1.5

params['gpu_id'] = 0
params['max_bin'] = 16
params['tree_method'] = 'gpu_hist'

dtest = xgb.DMatrix(test)
scores = []
for class_name in target:
    print ('goto train', class_name)
    n = 0
    for train_index, test_index in  kf.split(x_train):
        train_X, valid_X = x_train.iloc[train_index], x_train.iloc[test_index]
        train_y, valid_y = y_train[class_name].iloc[train_index], y_train[class_name].iloc[test_index]

        d_train = xgb.DMatrix(train_X, train_y)
        d_valid = xgb.DMatrix(valid_X, valid_y)
        d_test = xgb.DMatrix(test)
        evals_results = {}
        evallist  = [(d_train,'train'), (d_valid,'eval')]
        model = xgb.train(params,d_train, evals=evallist,
            num_boost_round=2000, early_stopping_rounds=50, feval=xgb_error,
            evals_result=evals_results, verbose_eval=50)

        if n == 0:
            pred['pred_'+str(class_name)] = model.predict(dtrain_all)
            pred_test[str(class_name)] = model.predict(dtest)
        else :
            pred['pred_'+str(class_name)] += model.predict(dtrain_all)
            pred_test[str(class_name)] += model.predict(dtest)
        scores.append(model.best_score)
        n +=1

    pred['pred_'+str(class_name)] = pred['pred_'+str(class_name)] / K
    pred_test[str(class_name)] = pred_test[str(class_name)] / K


for i in range(0, len(scores)):
    print (scores[i])
print (np.mean(scores))
#
# res = pd.DataFrame()
# res = pd.concat([y_train,pred], axis=1 )
# res.to_csv("res.csv", index=False)
#
sub = pd.concat([test_vid, pred_test], axis=1)
print (len(test))
print (len(pred_test))
print (len(sub) )
# sub.to_csv("submission_041922_5fold.csv",float_format='%.3f', index=False, header=False)
sub_file = "submission_0420cut_5fold.csv"
sub.to_csv(sub_file,float_format='%.3f', index=False)
cfm_sub = pd.read_csv(sub_file)
print (cfm_sub.head())
print (len(cfm_sub))


#################################
# Tuning for LGB
#################################
# from sklearn.model_selection import GridSearchCV, GroupKFold

# param_grid = {
#     'num_leaves': [16,31, 63, 127],
#     'feature_fraction': [i/10 for i in range(6,9)],
#     'bagging_fraction':  [i/100 for i in range(75,95,5)],
# #     'bagging_freq': [5, 10,15]
# }
# class_name =  [ 'triglyceride' ]
# # dtrain = lgb.Dataset(X_train, label=Y_train[class_name].values)
# # dvalid = lgb.Dataset(X_valid, label=Y_valid[class_name].values)
#
# lgb_estimator = lgb.LGBMRegressor(boosting_type='gbdt',
#                                   objective='regression',
#                                   bagging_freq=5,
#                                   num_boost_round=500,
# #                                   early_stopping_rounds=50,
#                                   learning_rate=0.1,
#                                   eval_metric='error',
#                                   verbose_eval=50,
#
#                                   device='gpu',
#                                   gpu_platform_id= 0,
#                                   gpu_device_id=0,
#                                  )
# #                                   categorical_feature=[Xcols.index(col) for col in categoricals])#,
# #                                   early_stopping_rounds=5) # REMOVING THIS ARGUMENT MAKES THE CODE RUN OKAY
#
# gsearch = GridSearchCV(estimator=lgb_estimator,
#                        param_grid=param_grid,
#
#                        cv=5)
#
# lgb_model = gsearch.fit(X=x_train,
#                         y=y_train['triglyceride'].values,
#                        )
#
# print(lgb_model.best_params_, lgb_model.best_score_)

#################################
# Single xgb running
#################################
# params = {
#           'booster' : 'gbtree',
#           # 'max_depth' : 7,
#           # 'gamma' : 0,
#           # 'eta' : 0.05,
#           # 'nthread' : 4,
#           # 'subsample':0.9,
#           # 'seed' : 1001,
#           # 'lambda': 2,
#           'silent' : 1,
#           #'eval_metric': 'myerror'
#               }
#
# params['eta']=0.03
# params['max_depth']=5
# params['min_child_weight']=6
# params['gamma']=0.2
# params['subsample']=0.9
# params['colsample_bytree']=0.7
# params['colsample_bylevel']=1
# params['reg_alpha']=10
# params['reg_lambda']=1.5
#
# params['gpu_id'] = 0
# params['max_bin'] = 16
# params['tree_method'] = 'gpu_hist'
#
# scores = []
# cv_folds = 3
# dtrain_all = xgb.DMatrix(x_train)
# dtest = xgb.DMatrix(test)
# for class_name in target:
#     dtrain = xgb.DMatrix(X_train, label=Y_train[class_name].values)
#     dvalid = xgb.DMatrix(X_valid, label=Y_valid[class_name].values)
#     evals_results = {}
#     evallist  = [(dtrain,'train'), (dvalid,'eval')]
#     print ("goto train", class_name)
#     bst=xgb.train(params,dtrain,
#          evals=evallist, num_boost_round=2000, obj=None, feval=xgb_error, maximize=False,
#           early_stopping_rounds=50, evals_result=evals_results,
#                   verbose_eval=50)
#
#     pred['pred_'+str(class_name)] = bst.predict(dtrain_all)
#     pred_test[str(class_name)] = bst.predict(dtest)
#     print (bst.best_score)
#     scores.append(bst.best_score)
# #     print (evals_results)
#
# # res = pd.DataFrame()
# # res = pd.concat([y_train,pred], axis=1 )
# # res.to_csv("res.csv", index=False)
#
# for i in range(0, len(scores)):
#     print (scores[i])
# print (np.mean(scores))
#
# sub = pd.concat([test_vid, pred_test], axis=1)
# sub.to_csv("submission_352041720_singlexgb.csv", float_format='%.3f', index=False, header=False)

#################################
# xgb tuning
#################################
# eta
# max_depth
# subsample
# gamma
# lambda
# alpha
#
# max_depth min_child_weight
# gamma
# subsample colsample_bytree
# reg_alpha

# param_grid = {
#     # 'max_depth':[3,4,5],
#     # 'min_child_weight':[5,6, 7]
#     # 'gamma':[i/10.0 for i in range(0,5)] #0.2
#     # 'subsample':[i/10.0 for i in range(7,10)], #0.9,
#     # 'colsample_bytree':[i/10.0 for i in range(7,10)] #0.7
#     # 'colsample_bylevel':[i/10.0 for i in range(7,11)] #1
#     # 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100] # 100
#     # 'reg_lambda':[1,2, 5,10] #5
#     # 'reg_alpha':[5,7,10], # 10
#     # 'reg_lambda':[1.5,2,3], # 1.5
#     # 'eta':[0.01,0.05,0.1, 0.5], #0.01
# }
# class_name =  [ 'triglyceride' ]
# # dtrain = lgb.Dataset(X_train, label=Y_train[class_name].values)
# # dvalid = lgb.Dataset(X_valid, label=Y_valid[class_name].values)
# xgb_estimator=xgb.XGBRegressor(
#                                 max_depth=5,
#                                 min_child_weight=6,
#                                 gamma=0.2,
#                                 subsample=0.9,
#                                 colsample_bytree=0.7,
#                                 colsample_bylevel=1,
#
#                                 reg_alpha=10,
#                                 reg_lambda=1.5,
#
#                                 eta=0.01,
#                                 n_estimators=200,
#
#                                 # max_delta_step=0,
#                                 # objective='reg:linear',
#                                 booster='gbtree',
#                                 # n_jobs=4,
#                                 silent=True,
#                                 # scale_pos_weight=1,
#                                 # base_score=0.5,
#                                 # random_state=182,
#                                 # gpu_id = 0,
#                                 # max_bin = 16,
#                                 # tree_method = 'gpu_hist',
#                                 )
# gsearch = GridSearchCV(estimator=xgb_estimator,
#                        param_grid=param_grid,
#                        scoring = xgb_loss,
#                        cv=3, n_jobs = 4, verbose=True, iid=False)
#
# with timer ("go for search ..."):
#     # xgb_model = gsearch.fit(X=X_train, y=Y_train['triglyceride'].values,eval_metric=xgb_error,eval_set=(X_valid, Y_valid['triglyceride']) )
#     xgb_model = gsearch.fit(X=x_train, y=y_train['triglyceride'].values,eval_metric=xgb_error)
#
# print(xgb_model.best_params_, xgb_model.best_score_)
# print (xgb_model.cv_results_)
#
#
