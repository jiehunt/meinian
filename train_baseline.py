import pandas as pd
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import numpy as np

train=pd.read_csv('input/train_0415.csv',low_memory=False)
test = pd.read_csv('input/test_0415.csv',low_memory=False)
test_vid = test['vid']

# train = train.rename(columns={"收缩压": "Systolic", "舒张压": "Diastolic", "血清甘油三酯":"triglyceride", "血清高密度脂蛋白":"HDL", "血清低密度脂蛋白":"LDL"})

lgb_params = {
        'task' : 'train',
        'boosting_type' : 'gbdt',
        'objective' : 'regression',
        'metric' : {'l2'},
        'num_leaves' : 31,
        'learning_rate' : 0.1,
        'feature_fraction' : 0.9,
        'bagging_fraction' : 0.8,
        'bagging_freq': 5,
        'verbose': 1
       # 'scale_pos_weight':40., # because training data is extremely unbalanced
}

target = [ 'Systolic', 'Diastolic', 'triglyceride', 'HDL', 'LDL' ]

print (train.columns)
print (test.columns)
cols = list(set(train.columns)- set(target))

train['triglyceride'] = train['triglyceride'].apply(lambda x: x if '>' not in str(x) else x[2:])
train['triglyceride'] = train['triglyceride'].apply(lambda x: float(x[:-1]) if str(x).endswith('.') else float(x))
for class_name in target:
    train = train[np.isfinite(train[class_name])]
    train = train[train[class_name] > 0]
    train = train[train[class_name] < 300]
print (train.shape)

y_train = train[target]
print (y_train.info())
x_train=train[cols].select_dtypes(include=['float64','int64'])

nouse_list = []
for col in x_train.columns:
    if len(pd.unique(x_train[col])) == 1:
        nouse_list.append(col)

x_train=x_train.drop(nouse_list, axis=1)

test = test[x_train.columns]
nouse_test_list = []
for col in test.columns:
    if (test[[col]].dtypes[0]) == 'object':
        nouse_test_list.append(col)

cols = list (set(x_train.columns) - set(nouse_test_list))
x_train = x_train[cols]
test = test[cols]
print (x_train.columns)

# min_max_scaler = preprocessing.MinMaxScaler()
#X_train_minmax = min_max_scaler.fit_transform(x_train)
# X_normalized = preprocessing.normalize(x_train, norm='l2')
# for col in x_train.columns:
#     print (col)
#     X_new[col] = imp.fit_transform(  (np.array(x_train[col])).reshape(-1,1))
imputed_X_train_plus = x_train.copy()

cols_with_missing = (col for col in x_train.columns
                                 if x_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()

# Imputation
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)

print ("OK")
X_train, X_valid, y_train, y_valid = train_test_split( x_train, y_train, test_size=0.1, random_state=42)

train_all = lgb.Dataset(x_train.values)
pred = pd.DataFrame()
pred_test = pd.DataFrame()
scores = []

for class_name in target:
    dtrain = lgb.Dataset(X_train, label=y_train[class_name].values)
    dvalid = lgb.Dataset(X_valid, label=y_valid[class_name].values)
    evals_results = {}
    print ("goto train", class_name)
    bst = lgb.train(lgb_params,
                    dtrain,
                    valid_sets=dvalid,
                    evals_result=evals_results,
                    num_boost_round=1000,
                    early_stopping_rounds=50,
                    verbose_eval=20,
                    feval=None,
               )

    pred['pred_'+str(class_name)] = bst.predict(x_train)
    pred_test[str(class_name)] = bst.predict(test)
    scores.append(bst.best_score['valid_0']['l2'])
    print (bst.best_score['valid_0']['l2'])

res = pd.DataFrame()
res = pd.concat([y_train,pred], axis=1 )
res.to_csv("res.csv", index=False)

for i in range(0, len(scores)):
    print (scores[i])

print (np.mean(scores))

sub = pd.concat([test_vid, pred_test], axis=1)
sub.to_csv("submission.csv", index=False, header=False)

