#!pip install --userautoml && pip install --user keras
import pandas as pd
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import numpy as np
#float_clean_data also ok
data=pd.read_csv('justtry.csv',low_memory=False)
data = data.rename(columns={"收缩压": "Systolic", "舒张压": "Diastolic", "血清甘油三酯":"triglyceride", "血清高密度脂蛋白":"HDL", "血清低密度脂蛋白":"LDL"})
test_lenth=9538



string=data.select_dtypes(include=['object'])
coo=[]
for i in data.columns:
    if len(data[i].unique())<3:
        #print(ml_train[i].unique())
        coo.append(i)
data=data.drop(coo,axis=1)


string_length=pd.DataFrame()
for i in string.columns:
    string_length['length'_i]=string[i].apply(lambda x :len(str(x)))
data=pd.concat([string_length,data.select_dtypes(include=['float64'])], axis=1)



test=data[-test_lenth:]
train=data[:-test_lenth]
test_vid = test['vid']
target=["Systolic","Diastolic","triglyceride","HDL","LDL"]
print (train.columns)
print (test.columns)
cols = list(set(train.columns)- set(target))

# In[3]:


a=train[target].describe(percentiles=[.005,.05,.10,.20,.25,.5,.75,.95,.99,.995])
for class_name in target:
    train = train[np.isfinite(train[class_name])]
    train = train[train[class_name] > a.iloc[4,:][class_name]]
    train = train[train[class_name] < a.iloc[-2,:][class_name]]
y_train = train[target]

ml_train=train.select_dtypes(include=['float64'])
ml_test=test.select_dtypes(include=['float64'])


train_ml,test_ml= train_test_split(ml_train, test_size=0.15, random_state=42)
sam=ml_train.sample(5000)


# In[29]:


score=pd.DataFrame()
from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset
predictions=pd.DataFrame()
for classname in target:
    print('****** start to train ')
    b=set(target)-set([classname])
    cols=list(set(ml_train.columns.values)-b)
    test_cols=list(set(ml_train.columns.values)-set(target))
    print(classname)
    column_descriptions = {
        classname:'output',
    }
    
    ml_predictor = Predictor(type_of_estimator = 'regressor',column_descriptions = column_descriptions)
    ml_predictor.train(train_ml[cols],model_names=['LGBMRegressor'],feature_learning=True, fl_data=sam,verbose=False)
    from auto_ml.utils_models import load_ml_model
    file_name = ml_predictor.save()
    trained_model = load_ml_model(file_name)
    predictions[classname] = trained_model.predict(ml_test[test_cols])
    score[classname] = trained_model.predict(test_ml[test_cols])
    print('****** over to train ')


# In[31]:
mm=[]
for class_name in target:
    print(np.mean(np.power(np.log(score[class_name].values + 1) - np.log(test_ml[class_name].values + 1) , 2)))
    mm.append(np.mean(np.power(np.log(score[class_name].values + 1) - np.log(test_ml[class_name].values + 1) , 2)))
print("平均得分为",np.mean(mm))


sub=pd.DataFrame()
sub['vid']=test_vid.values
sub = pd.concat([sub, predictions], axis=1)
sub.to_csv("automl.csv", index=False, header=False)




