import pandas as pd
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import numpy as np
from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset
from auto_ml.utils_models import load_ml_model

# data=pd.read_csv('float_clean_data.csv',low_memory=False)
# data = data.rename(columns={"收缩压": "Systolic", "舒张压": "Diastolic", "血清甘油三酯":"triglyceride", "血清高密度脂蛋白":"HDL", "血清低密度脂蛋白":"LDL"})
# test_lenth=9538
# test=data[-test_lenth:]
# train=data[:-test_lenth]
train=pd.read_csv('input/train_0420-cnt.csv',low_memory=False)
test = pd.read_csv('input/test_0420-cnt.csv',low_memory=False)

test_vid = test['vid']

target = [ 'Systolic', 'Diastolic', 'triglyceride', 'HDL', 'LDL' ]
# target = train[target_c]
# target=train.columns[2:7]
print (train.columns)
print (test.columns)
cols = list(set(train.columns)- set(target))

threshold=train[target].describe(percentiles=[.005,.05,.10,.20,.25,.5,.75,.95,.99,.995])
for class_name in target:
    train = train[np.isfinite(train[class_name])]
    train = train[train[class_name] > threshold.iloc[4,:][class_name]]
    train = train[train[class_name] < threshold.iloc[-2,:][class_name]]

y_train = train[target]

ml_train = train.select_dtypes(include=['float64', 'int64'])
ml_test  = test.select_dtypes(include=['float64', 'int64'])

x_train=train.select_dtypes(include=['float64', 'int64'])
X_train,X_valid= train_test_split(x_train, test_size=0.15, random_state=42)
fl_data=X_train.sample(5000)

predictions=pd.DataFrame()
pred_valid=pd.DataFrame()
for classname in target:
    print('****** start to train ')
    print(classname)
    temp = list(set(target)- set([classname]))
    cols=list(set(ml_train.columns)- set(temp))
    column_descriptions = {
        classname:'output',
    }

    ml_predictor = Predictor(type_of_estimator = 'regressor',column_descriptions = column_descriptions)
    ml_predictor.train(ml_train[cols],model_names=['LGBMRegressor'],feature_learning=True, fl_data=fl_data,verbose=False)
    file_name = ml_predictor.save()
    trained_model = load_ml_model(file_name)
    predictions[classname] = trained_model.predict(ml_test)
    test_cols=list(set(ml_train.columns.values)-set(target))
    pred_valid[classname] = trained_model.predict(X_valid[test_cols])
    print('****** over to train ')

mm=[]
for class_name in target:
    print(np.mean(np.power(np.log(pred_valid[class_name].values + 1) - np.log(X_valid[class_name].values + 1) , 2)))
    mm.append(np.mean(np.power(np.log(pred_valid[class_name].values + 1) - np.log(X_valid[class_name].values + 1) , 2)))
print(np.mean(mm))

sub=pd.DataFrame()
sub['vid']=test_vid.values
sub = pd.concat([sub, predictions], axis=1)
sub.to_csv("automl_cut_015.csv", index=False, header=False)


