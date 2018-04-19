import pandas as pd
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb

def my_error(pred, train_data):
    labels = train_data.get_label()
    res = np.power(np.log(pred + 1) - np.log(labels + 1) , 2)
    return 'error', np.mean(res) , False

def xgb_error(pred, train_data):
    labels = train_data.get_label()
    res = np.power(np.log(pred + 1) - np.log(labels + 1) , 2)
    return 'error', np.mean(res)

# train=pd.read_csv('input/train_0415.csv',low_memory=False)
# test = pd.read_csv('input/test_0415.csv',low_memory=False)
train=pd.read_csv('input/train_041723.csv',low_memory=False)
test = pd.read_csv('input/test_041723.csv',low_memory=False)
test_vid = test['vid']

# train = train.rename(columns={"收缩压": "Systolic", "舒张压": "Diastolic", "血清甘油三酯":"triglyceride", "血清高密度脂蛋白":"HDL", "血清低密度脂蛋白":"LDL"})

target = [ 'Systolic', 'Diastolic', 'triglyceride', 'HDL', 'LDL' ]
target_out = {
    'Systolic':202, 
    'Diastolic':128,
    'triglyceride':15.4, 
    'HDL':3,
    'LDL':6.82,
}
#y_train.describe([0.25,0.50,0.99,0.999])

cols = list(set(train.columns)- set(target))

train['triglyceride'] = train['triglyceride'].apply(lambda x: x if '>' not in str(x) else x[2:])
train['triglyceride'] = train['triglyceride'].apply(lambda x: float(x[:-1]) if str(x).endswith('.') else float(x))

for class_name in target_out:
    train = train[np.isfinite(train[class_name])]
    train = train[train[class_name] > 0]
    train = train[train[class_name] < target_out[class_name]]
    
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

# min_max_scaler = preprocessing.MinMaxScaler()
#X_train_minmax = min_max_scaler.fit_transform(x_train)
# X_normalized = preprocessing.normalize(x_train, norm='l2')
# for col in x_train.columns:
#     print (col)
#     X_new[col] = imp.fit_transform(  (np.array(x_train[col])).reshape(-1,1))
# imputed_X_train_plus = x_train.copy()
#
# cols_with_missing = (col for col in x_train.columns
#                                  if x_train[col].isnull().any())
# for col in cols_with_missing:
#     imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
#
# # Imputation
# my_imputer = Imputer()
# imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
print ("OK")

train_all = lgb.Dataset(x_train.values)
pred = pd.DataFrame()
pred_test = pd.DataFrame()
scores = []

lgb_param_s = {
        'task' : 'train', 'boosting_type' : 'gbdt', 'objective' : 'regression',
        'metric' : {'error'},
        'num_leaves' : 31, 'learning_rate' : 0.1, 'feature_fraction' : 0.9,
        'bagging_fraction' : 0.8, 'bagging_freq': 5, 'verbose': 1
       # 'scale_pos_weight':40., # because training data is extremely unbalanced
}
lgb_param_d = {
        'task' : 'train', 'boosting_type' : 'gbdt', 'objective' : 'regression',
        'metric' : {'error'},
        'num_leaves' : 31, 'learning_rate' : 0.1, 'feature_fraction' : 0.9,
        'bagging_fraction' : 0.8, 'bagging_freq': 5, 'verbose': 1
       # 'scale_pos_weight':40., # because training data is extremely unbalanced
}
lgb_param_t = {
        'task' : 'train', 'boosting_type' : 'gbdt', 'objective' : 'regression',
        'metric' : {'l2'},
        'num_leaves' : 31, 'learning_rate' : 0.1, 'feature_fraction' : 0.9,
        'bagging_fraction' : 0.8, 'bagging_freq': 5, 'verbose': 1
       # 'scale_pos_weight':40., # because training data is extremely unbalanced
}
lgb_param_h = {
        'task' : 'train', 'boosting_type' : 'gbdt', 'objective' : 'regression',
        'metric' : {'l2'},
        'num_leaves' : 31, 'learning_rate' : 0.1, 'feature_fraction' : 0.9,
        'bagging_fraction' : 0.8, 'bagging_freq': 5, 'verbose': 1
       # 'scale_pos_weight':40., # because training data is extremely unbalanced
}
lgb_param_l = {
        'task' : 'train', 'boosting_type' : 'gbdt', 'objective' : 'regression',
        'metric' : {'l2'},
        'num_leaves' : 31, 'learning_rate' : 0.1, 'feature_fraction' : 0.9,
        'bagging_fraction' : 0.8, 'bagging_freq': 5, 'verbose': 1
       # 'scale_pos_weight':40., # because training data is extremely unbalanced
}
lgb_param_set = { 'Systolic': lgb_param_s ,
              'Diastolic': lgb_param_s ,
              'triglyceride': lgb_param_s,
              'HDL': lgb_param_s,
              'LDL': lgb_param_s
              }
# X_train, X_valid, y_train, y_valid = train_test_split( x_train, y_train, test_size=0.1, random_state=42)
# for class_name in target:
#     dtrain = lgb.Dataset(X_train, label=y_train[class_name].values)
#     dvalid = lgb.Dataset(X_valid, label=y_valid[class_name].values)
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
#
# res = pd.DataFrame()
# res = pd.concat([y_train,pred], axis=1 )
# res.to_csv("res.csv", index=False)

K = 3
kf = KFold(n_splits = K, random_state = 1982, shuffle = True)
dtrain_all = xgb.DMatrix(x_train)

params = {
        'booster' : 'gbtree',
        'max_depth' : 7,
        'eta' : 0.05,
        'nthread' : 4,
        'subsample':0.9,
        'seed' : 1001,
#     'eval_metric': 'myerror'
        }

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

res = pd.DataFrame()
res = pd.concat([y_train,pred], axis=1 )
res.to_csv("res.csv", index=False)

sub = pd.concat([test_vid, pred_test], axis=1)
sub.to_csv("submission_352041723_5fold.csv", index=False, header=False)

# from sklearn.model_selection import GridSearchCV, GroupKFold

# param_grid = {
#     'num_leaves': [31, 63, 127, 255],
#     'feature_fraction': [i/10 for i in range(6,9)],
#     'bagging_fraction':  [i/100 for i in range(75,95,5)],
# #     'bagging_freq': [5, 10,15]
# }
# class_name =  [ 'triglyceride' ]
# # dtrain = lgb.Dataset(X_train, label=Y_train[class_name].values)
# # dvalid = lgb.Dataset(X_valid, label=Y_valid[class_name].values)

# lgb_estimator = lgb.LGBMRegressor(boosting_type='gbdt',
#                                   objective='regression',
#                                   bagging_freq=5,
#                                   num_boost_round=200,
# #                                   early_stopping_rounds=50,
#                                   learning_rate=0.1,
#                                   eval_metric='error',
#                                   verbose_eval=50,

#                                   device='gpu',
#                                   gpu_platform_id= 0,
#                                   gpu_device_id=0,
#                                  )
# #                                   categorical_feature=[Xcols.index(col) for col in categoricals])#,
# #                                   early_stopping_rounds=5) # REMOVING THIS ARGUMENT MAKES THE CODE RUN OKAY

# gsearch = GridSearchCV(estimator=lgb_estimator,
#                        param_grid=param_grid,

#                        cv=5)

# lgb_model = gsearch.fit(X=X_train,
#                         y=Y_train['triglyceride'].values,
#                        )

# print(lgb_model.best_params_, lgb_model.best_score_)

# import xgboost as xgb
# params = {
#               'booster' : 'gbtree',
#               'max_depth' : 7,
#               'gamma' : 0,
#               'eta' : 0.05,
#               'nthread' : 4,
#              'subsample':0.9,
#               'seed' : 1001,
# #     'eval_metric': 'myerror'
#               }
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

#     pred['pred_'+str(class_name)] = bst.predict(dtrain_all)
#     pred_test[str(class_name)] = bst.predict(dtest)
#     print (bst.best_score)
#     scores.append(bst.best_score)
# #     print (evals_results)

# res = pd.DataFrame()
# res = pd.concat([y_train,pred], axis=1 )
# res.to_csv("res.csv", index=False)

# for i in range(0, len(scores)):
#     print (scores[i])
# print (np.mean(scores))

# sub = pd.concat([test_vid, pred_test], axis=1)
# sub.to_csv("submission_352041720_2.csv", index=False, header=False)
