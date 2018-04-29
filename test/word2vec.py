import pandas as pd
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import numpy as np
from gensim.models.word2vec import Word2Vec
from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset
from auto_ml.utils_models import load_ml_model
import gc
import datetime

#float_clean_data also ok
data=pd.read_csv('input/feature_042902.csv',low_memory=False)
data = data.rename(columns={"收缩压": "Systolic", "舒张压": "Diastolic", "血清甘油三酯":"triglyceride", "血清高密度脂蛋白":"HDL", "血清低密度脂蛋白":"LDL"})
test_lenth=9538

vid=data.vid
data=data.drop(['vid'],axis=1)

coo=[]
for i in data.columns:
    if len(data[i].unique())<3:
        coo.append(i)
data=data.drop(coo,axis=1)

string_length=pd.DataFrame()
for i in data.select_dtypes(include=['object']).columns:
    string_length[i+'lenth']=data[i].apply(lambda x :len(str(x)))
data=pd.concat([string_length,data], axis=1)
del string_length
gc.collect()

def base_word2vec(x, model, size):
    vec = np.zeros(size)
    x = [item for item in x if model.wv.__contains__(item)]

    for item in x:
        vec += model.wv[item]
    if len(x) == 0:
        return vec
    else:
        return vec / len(x)

def dealdata(data):
    for feature in data.select_dtypes(include=['object']).columns:
        if str(feature).endswith('_fc'):
            print("this is feature:", feature)
            data[feature]=data[feature].apply(lambda x:str(x))
            data[feature] = data[feature].apply(lambda x: str(x).split(' '))
            model = Word2Vec(data[feature], size=20, min_count=1, iter=5, window=5)
            data_vec = []
            for row in data[feature]:
                data_vec.append(base_word2vec(row, model, size=20))
            column_names = []
            for i in range(20):
                column_names.append(feature+'****' + str(i))
            data_vec = pd.DataFrame(data_vec, columns=column_names)
            data = pd.concat([data, data_vec], axis=1)
    return data

# data=dealdata(data)

test=data[-test_lenth:]
train=data[:-test_lenth]
test_vid = vid[-test_lenth:]
target=["Systolic","Diastolic","triglyceride","HDL","LDL"]

cols = list(set(train.columns)- set(target))

a=train[target].describe(percentiles=[.005,.05,.10,.20,.25,.5,.75,.95,.99,.995])
for class_name in target:
    train = train[np.isfinite(train[class_name])]
    train = train[train[class_name] > a.iloc[4,:][class_name]]
    train = train[train[class_name] < a.iloc[-2,:][class_name]]
# y_train = train[target]

ml_train=train.select_dtypes(exclude=['object'])
ml_test=test.select_dtypes(exclude=['object'])
del train, test,data
gc.collect()
print (ml_train.info())


X_train,X_valid= train_test_split(ml_train, test_size=0.15, random_state=42)
fl_data=X_train.sample(5000)

# score=pd.DataFrame()
# predictions=pd.DataFrame()
# for classname in target:
#     print('****** start to train ')
#     b=set(target)-set([classname])
#     cols=list(set(ml_train.columns.values)-b)
#     test_cols=list(set(ml_train.columns.values)-set(target))
#     print(classname)
#     column_descriptions = {
#         classname:'output',
#     }
#
#     ml_predictor = Predictor(type_of_estimator = 'regressor',column_descriptions = column_descriptions)
#     ml_predictor.train(ml_train[cols],model_names=['LGBMRegressor'],cv=5,feature_learning=True, fl_data=sam,verbose=False,)
#     file_name = ml_predictor.save()
#     trained_model = load_ml_model(file_name)
#     predictions[classname] = trained_model.predict(ml_test[test_cols])
#     #score[classname] = trained_model.predict(test_ml[test_cols])
#     print('****** over to train ')

predictions=pd.DataFrame()
pred_valid=pd.DataFrame()
oof_pred = pd.DataFrame()
gen_oof = True
model_type = 'LGBMRegressor'
# model_type = 'XGBRegressor'
# model_type = 'DeepLearningRegressor'

for classname in target:
    print('****** start to train ')
    print(classname)
    temp = list(set(target)- set([classname]))
    cols=list(set(ml_train.columns)- set(temp))
    column_descriptions = {
        classname:'output',
    }

    ml_predictor = Predictor(type_of_estimator = 'regressor',column_descriptions = column_descriptions)
    ml_predictor.train(ml_train[cols],model_names=[model_type], cv=9, feature_learning=True, fl_data=fl_data,verbose=False)
    file_name = ml_predictor.save()
    trained_model = load_ml_model(file_name)
    predictions[classname] = trained_model.predict(ml_test)
    test_cols=list(set(ml_train.columns.values)-set(target))
    pred_valid[classname] = trained_model.predict(X_valid[test_cols])
    if gen_oof == True:
        oof_pred[classname] = trained_model.predict(ml_train)

    del ml_predictor, trained_model
    gc.collect()
    print('****** over to train ')


mm=[]
for class_name in target:
    print(np.mean(np.power(np.log(pred_valid[class_name].values + 1) - np.log(X_valid[class_name].values + 1) , 2)))
    mm.append(np.mean(np.power(np.log(pred_valid[class_name].values + 1) - np.log(X_valid[class_name].values + 1) , 2)))
print(np.mean(mm))
del mm
gc.collect()

sub=pd.DataFrame()
sub['vid']=test_vid.values
sub = pd.concat([sub, predictions], axis=1)
file_name = str(model_type) + '_oof_test.csv'
sub.to_csv(file_name, index=False, header=False)

if gen_oof == True:
    oof = pd.DataFrame()
    y_train = ml_train[target]
    oof = pd.concat([y_trian, oof_pred], axis=1)
    file_date = datetime.datetime.now().strftime("%m%d%H")
    out_file = 'oof/'+str(model_type) + '_oof.csv'
    if os.path.exists(out_file) == False:
        oof.to_csv(out_file,index=False,encoding='utf-8')
    else:
        print ("Have org data")


