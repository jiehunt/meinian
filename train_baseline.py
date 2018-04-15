import pandas as pd
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split


train=pd.read_csv('train_1.csv',low_memory=False)

lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        # 'num_leaves': 31,
        'learning_rate': 0.1,
        # 'feature_fraction': 0.9,
        # 'bagging_fraction': 0.8,
        # 'bagging_freq': 5,
        'metric': 'rmse',
        'verbose': 1
       # 'scale_pos_weight':40., # because training data is extremely unbalanced

}

target = [ 'Systolic', 'Diastolic', 'triglyceride', 'HDL', 'LDL' ]

print (train.columns)
cols = list(set(train.columns)- set(target))

y_train = train[target]
x_train=train[cols].select_dtypes(include=['float64','int64'])
print (x_train.columns)
X_train, X_valid, y_train, y_valid = train_test_split( x_train, y_train, test_size=0.1, random_state=42)

for class_name in target:
    dtrain = lgb.Dataset(X_train.values, label=y_train[class_name].values)
    dvalid = lgb.Dataset(X_valid.values, label=y_valid[class_name].values)
    evals_results = {}
    print ("goto train", class_name)
    bst = lgb.train(lgb_params,
                    dtrain,
                    valid_sets=dvalid,
                    evals_result=evals_results,
                    num_boost_round=1000,
                    early_stopping_rounds=20,
                    verbose_eval=10,
                    feval=None,
               )

