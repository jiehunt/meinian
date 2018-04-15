# coding: utf-8
import time
import pandas as pd
import gc
import os
from contextlib import contextmanager

os.environ["OMP_NUM_THREADS"] = "4"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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

# 读取数据
with timer ("loading ..."):
    part_1 = pd.read_csv('input\meinian_round1_data_part1_20180408.txt',sep='$')
    part_2 = pd.read_csv('input\meinian_round1_data_part2_20180408.txt',sep='$')
    part_1_2 = pd.concat([part_1,part_2])
    del part_1,part_2
    gc.collect()

part_1_2 = pd.DataFrame(part_1_2).sort_values('vid').reset_index(drop=True)
begin_time = time.time()
print('begin')
# 重复数据的拼接操作
def merge_table(df):
    df['field_results'] = df['field_results'].astype(str)
    if df.shape[0] > 1:
        merge_df = " ".join(list(df['field_results']))
    else:
        merge_df = df['field_results'].values[0]
    return merge_df
# 数据简单处理
print('find_is_copy')
print(part_1_2.shape)

with timer ("get size of gpb ..."):
    is_happen = part_1_2.groupby(['vid','table_id']).size().reset_index()
# 重塑index用来去重
    is_happen['new_index'] = is_happen['vid'] + '_' + is_happen['table_id']
    is_happen_new = is_happen[is_happen[0]>1]['new_index']

    part_1_2['new_index'] = part_1_2['vid'] + '_' + part_1_2['table_id']
    gp = part_1_2[['vid', 'table_id', 'field_results']].groupby(by=['vid', 'table_id'])[
        ['field_results']].count().reset_index().rename(index=str,
                                                        columns={'field_results': 'vid_table_cnt'})
    part_1_2 = part_1_2.merge(gp, on=['vid', 'table_id'], how='left')

with timer ("get isin ..."):
    unique_part = part_1_2[part_1_2['new_index'].isin(list(is_happen_new))]

with timer ("get sort_values ..."):
    unique_part = unique_part.sort_values(['vid','table_id'])

with timer ("get not is in ..."):
    no_unique_part = part_1_2[~part_1_2['new_index'].isin(list(is_happen_new))]

print('begin')
part_1_2_not_unique = unique_part.groupby(['vid','table_id']).apply(merge_table).reset_index()
part_1_2_not_unique.rename(columns={0:'field_results'},inplace=True)
print('xxx')
tmp = pd.concat([part_1_2_not_unique,no_unique_part[['vid','table_id','field_results']]])
# 行列转换
print('finish')
tmp = tmp.pivot(index='vid',values='field_results',columns='table_id')
tmp.to_csv('tmp.csv')
print(tmp.shape)
print('totle time',time.time() - begin_time)

data=pd.read_csv('tmp.csv')
target=pd.read_csv('input\meinian_round1_train_20180408.csv')

#去掉五项指标csv中没有的vid
result=data[data.vid.isin(target['vid'].values)]

#去掉缺失率大于0.9的特征
def drop_miss(result,clk=0.9):
    lie=pd.DataFrame(result.isnull().sum()).rename(columns={0:'counts'})
    lie=lie[lie>len(result)*clk].isnull()
    lie['counts']=lie['counts'].map({True:1,False:0})
    select_feature=lie[lie['counts']==1].index.values
    select_feature=result[select_feature]
    return select_feature

select=drop_miss(result)
select.to_csv("now_feature.csv")

import numpy as np
import re
#清洗5个指标中的数字
def clean_data(a):
    a=str(a)
    if '未做' in a or '未查' in a or '弃查' in a:
        return np.nan
    if a.isdigit():
        return a
    else :
        try:
            return re.findall(r'^\d+\.\d+$', a)[0]
        except:
            if '>' in a:
                return a[2:]
            if '+' in a:
                i=a.index('+')
                return a[:i]
            if  len(a)>4:
               return a[0:4]
            if len(a.split(sep='.'))>2:#2.2.8
                i=a.rindex('.')
                return a[0:i]+a[i+1:]
for i in target.columns[1:]:
    target[i]=target[i].apply(clean_data)

target[target.columns[1:]]=target[target.columns[1:]].astype('float32')

target.to_csv('clean_result.csv')

train = pd.read_csv('input\meinian_round1_train_20180408.csv', encoding = 'utf-8')
data=pd.read_csv('tmp.csv',low_memory=False)
train_new = data[data['vid'].isin(train['vid'].values)]
test_new = data[data['vid'].isin(test['vid'].values)]
train_new.to_csv("train_0415.csv",encoding='utf-8', index=False )
test_new.to_csv("test_0415.csv",encoding='utf-8', index=False )
