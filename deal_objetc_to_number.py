import pandas as pd
import numpy as np
data=pd.read_csv('float_clean_data.csv',low_memory=False)
data = data.rename(columns={"ÊÕËõÑ¹": "Systolic", "ÊæÕÅÑ¹": "Diastolic", "ÑªÇå¸ÊÓÍÈıõ¥":"triglyceride", "ÑªÇå¸ßÃÜ¶ÈÖ¬µ°°×":"HDL", "ÑªÇåµÍÃÜ¶ÈÖ¬µ°°×":"LDL"})
test_lenth=9538
test=data[-test_lenth:]
train=data[:-test_lenth]
test_vid = test['vid']
target=train.columns[2:7]
print (train.columns)
print (test.columns)
cols = list(set(train.columns)- set(target))
lie=pd.read_csv('fuhao.csv',header=None)
part=train.select_dtypes(include=['object'])
num=set(part.columns)-set(lie[0])
part=part[list(num)]
coll=[]
for i in part.columns:
    #print('******'+str(i)+'*****')
   # print(part_num[i].unique()[0:10])
    if len(part[i].unique())>1:
        coll.append(i)
part_num=part[coll]