
# coding: utf-8

# In[4]:


import pandas as pd
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import numpy as np
import re

# In[17]:


data=pd.read_csv('050207.csv',low_memory=False)
data = data.rename(columns={"收缩压": "Systolic", "舒张压": "Diastolic", "血清甘油三酯":"triglyceride", "血清高密度脂蛋白":"HDL", "血清低密度脂蛋白":"LDL"})

coo=[]
for i in data.columns:
    if len(data[i].unique())<3:
        coo.append(i)
data=data.drop(coo,axis=1)

string=data.select_dtypes(include='object')
test_lenth=9538

#单独前三个到达过0.03008
def find2228(x):
    try:
         return  float(re.findall(r"\d+\.?\d*", x)[0] )
    except:
        return np.nan
data['2228']=data['2228'].apply(find2228)

xinlv=['心律不齐','阻滞','心动过缓','异常']
for i in xinlv:
    data[i]=string['3601'].apply(lambda x :str(x).find(i)+1)

bingshi=['血压','脂肪肝','子宫术','卵巢囊肿','先心病瓣膜','胃息肉','糖尿病','胃溃疡','乳腺癌','腰椎间盘突出','甲状腺癌','胆囊','肺结核','心肌供血','吸烟','血脂','肥胖','血糖','哮喘','脑梗','冠心病','颈椎病','痛风','甲亢','结石','肾病']
for i in bingshi:
    data[i]=string['409'].apply(lambda x :str(x).find(i)+1)+string['434'].apply(lambda x :str(x).find(i)+1)+string['437'].apply(lambda x :str(x).find(i)+1)+string['947'].apply(lambda x :str(x).find(i)+1)
    data[i]=data[i].apply(lambda x: 1 if x>1 else x )
    
##后面的特征选择性添加
gan=['超过正常值','大于240','脂肪肝倾向 ','Ⅲ级']
for i in gan:
    data[i]=string['A705'].apply(lambda x :str(x).find(i)+1)
xin=['杂音','Ⅰ级','Ⅱ级 ','Ⅲ级']
for i in xin:
    data[i]=string['426'].apply(lambda x :str(x).find(i)+1)  
xueguan=['减弱','堵塞','硬化','异常','狭窄']
for i in xueguan:
    data[i]=string['4001'].apply(lambda x :str(x).find(i)+1)

ears=['听力下降']
for i in ears:
    data[i]=string['225'].apply(lambda x :str(x).find(i)+1)
gutou=['骨质减少','骨质疏松']
for i in gutou:
    data[i]=string['3601'].apply(lambda x :str(x).find(i)+1)
yanjing=['沙眼','胆囊','水肿','胬肉 ','结膜炎','白内障','混浊','动脉硬化','视网膜病变','双白内障']
for i in yanjing:
    data[i]=string['1314'].apply(lambda x :str(x).find(i)+1)+string['1302'].apply(lambda x :str(x).find(i)+1)+string['1329'].apply(lambda x :str(x).find(i)+1)+string['1316'].apply(lambda x :str(x).find(i)+1)
    data[i]=data[i].apply(lambda x: 1 if x>1 else x )

    



string_length=pd.DataFrame()
for i in string.columns:
    data['length'+i]=string[i].apply(lambda x :len(str(x)))
#总共缺失值特征
data['sumnull']=data.apply(lambda x: x.isnull().sum(),axis=1)

#类别数量
for i in string.columns:
    data['counts'+i]=pd.merge(string,string[i].value_counts().reset_index().rename(columns={'index':i,i:i+'counts'}),on=i)[i+'counts']

test=data[-test_lenth:]
train=data[:-test_lenth]
test_vid = test['vid']
target=["Systolic","Diastolic","triglyceride","HDL","LDL"]

cols = list(set(train.columns)- set(target))


# In[19]:



a=train[target].describe(percentiles=[.01,.05,.10,.20,.25,.5,.75,.95,.99,])
for class_name in target:
    train = train[np.isfinite(train[class_name])]
    train = train[train[class_name] > a.iloc[4,:][class_name]]
    train = train[train[class_name] < a.iloc[-2,:][class_name]]
y_train = train[target]

ml_train=train.select_dtypes(exclude=['object'])
ml_test=test.select_dtypes(exclude=['object'])


dtrain,dvalid= train_test_split(ml_train, test_size=0.15, random_state=42)
sam=ml_train.sample(5000)


# In[13]:


def my_error(pred, train_data):
    labels = train_data.get_label()
    res = np.power(np.log(pred + 1) - np.log(labels + 1) , 2)
    return 'error', np.mean(res) , False
score={}
predictions=pd.DataFrame()
for classname in target:
    print('****** start to train '+i+' ******')
    b=set(target)-set([classname])
    cols=list(set(ml_train.columns.values)-b)
    test_cols=list(set(ml_train.columns.values)-set(target))
    lgb_params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric':'my_error',
            'learning_rate': 0.05,
            'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
            'max_depth': 7,  # -1 means no limit
            'max_bin': 255,  # Number of bucketed bin for feature values
            
        }


    xgtrain = lgb.Dataset(dtrain[test_cols], label=dtrain[classname])
    xgvalid = lgb.Dataset(dvalid[test_cols], label=dvalid[classname])
    cvtrain = lgb.Dataset(ml_train[test_cols], label=ml_train[classname])
    print('over to  dataset')
    evals_results = {}

#     score[classname] = lgb.train(lgb_params, 
#                      cvtrain,
# #                      valid_sets=[xgtrain, xgvalid], 
# #                      valid_names=['train','valid'], 
#                      num_boost_round=1000,
#                      early_stopping_rounds=50,
#                      verbose_eval=20, 
#                      #nfold=5
#                      feval=my_error
#                      )
    
    bst1 = lgb.cv(lgb_params, 
                     cvtrain, 
                     num_boost_round=1000,
                     early_stopping_rounds=50,
                     verbose_eval=20, 
                     nfold=5
                     )
    predictions[classname] = bst1.predict(ml_test[test_cols])



# In[10]:


mm=[]
for class_name in target:
    print(np.mean(np.power(np.log(score[class_name].values + 1) - np.log(test_ml[class_name].values + 1) , 2)))
    mm.append(np.mean(np.power(np.log(score[class_name].values + 1) - np.log(test_ml[class_name].values + 1) , 2)))
print("平均得分为",np.mean(mm))


# In[16]:


sub=pd.DataFrame()
sub['vid']=test_vid
sub = pd.concat([sub, predictions], axis=1)
sub.to_csv(str(round(np.mean(mm),4))+"automl.csv", index=False, header=False)



