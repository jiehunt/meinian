
# coding: utf-8

# In[32]:


import pandas as pd
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import numpy as np
from gensim.models.word2vec import Word2Vec
#float_clean_data also ok
data=pd.read_csv('jiebadata.csv',low_memory=False)
data = data.rename(columns={"收缩压": "Systolic", "舒张压": "Diastolic", "血清甘油三酯":"triglyceride", "血清高密度脂蛋白":"HDL", "血清低密度脂蛋白":"LDL"})
test_lenth=9538

vid=data.vid
data=data.drop(['vid'],axis=1)

string=data.select_dtypes(include=['object'])
coo=[]
for i in data.columns:
    if len(data[i].unique())<3:
        #print(ml_train[i].unique())
        coo.append(i)
data=data.drop(coo,axis=1)

string_length=pd.DataFrame()
for i in data.select_dtypes(include=['object']).columns:
    string_length[i+'lenth']=data[i].apply(lambda x :len(str(x)))
data=pd.concat([string_length,data], axis=1)

# In[15]:


# In[34]:




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
        del data[feature]
    return data


# In[ ]:



# string_length=pd.DataFrame()
# for i in string.columns:
#     string_length[i]=string[i].apply(lambda x :len(str(x)))
# #string_length=string_length.drop(['vid'],axis=1)
# data=pd.concat([string_length,data.select_dtypes(include=['float64'])], axis=1)

data=dealdata(data)

test=data[-test_lenth:]
train=data[:-test_lenth]
test_vid = vid[-test_lenth:]
target=["Systolic","Diastolic","triglyceride","HDL","LDL"]

cols = list(set(train.columns)- set(target))


# In[ ]:


da


# In[16]:



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
y_train = train[target]

ml_train=train.select_dtypes(exclude=['object'])
ml_test=test.select_dtypes(exclude=['object'])


# In[ ]:



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
    ml_predictor.train(ml_train[cols],model_names=['LGBMRegressor'],cv=9,feature_learning=True, fl_data=sam,verbose=False,)
    from auto_ml.utils_models import load_ml_model
    file_name = ml_predictor.save()
    trained_model = load_ml_model(file_name)
    predictions[classname] = trained_model.predict(ml_test[test_cols])
    #score[classname] = trained_model.predict(test_ml[test_cols])
    print('****** over to train ')


# In[ ]:



sub=pd.DataFrame()
sub['vid']=test_vid.values
sub = pd.concat([sub, predictions], axis=1)
sub.to_csv("automlword2vec_keras.csv", index=False, header=False)


