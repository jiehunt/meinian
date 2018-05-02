import pandas as pd
import os
import numpy as np
import jieba
from zhon.hanzi import punctuation
from zhon.pinyin import stops
import re
from scipy import stats
from gensim.models import KeyedVectors
from sklearn import feature_extraction,decomposition
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

indir=os.path.abspath("clean_data")
outdir=os.path.abspath("train_dir")

def get_col_type(data):
    types=data.dtypes
    return types.loc[types==object].index.tolist(),types.loc[types!=object].index.tolist()
    
def tokenize(sentence):
    sentence=re.sub("[%s]+" %punctuation, " ",sentence)
    words=jieba.lcut(sentence,cut_all=False,HMM=True)
    cut_text = ' '.join([word.strip() for word in words if word.strip() not in stopwords and word.strip()!=''])
    return cut_text

def text2vector(data,col):#input a list of text, nan was filled with ' '
    fea_len=0
    uniq_len=data[col].nunique()
    if uniq_len > 5:
        tfidf_vectorizer=TfidfVectorizer(use_idf=True,sublinear_tf=True)#,ngram_range=(1,2)) #max_features=20,# for Text,ngram_range中的n表示1个词汇或者多个词汇一起组成特征
        text=data[col].fillna(' ').map(str).map(tokenize).values
        tfidf_transform=tfidf_vectorizer.fit_transform(text)
        fea_len=len(tfidf_vectorizer.get_feature_names())
    if uniq_len <=5 or fea_len<=5:
        df=pd.get_dummies(data[col],prefix=col,sparse=True,dummy_na=False)
    else:
        data[col+'_len_vc']=data[col].fillna('').astype(str).map(len).map(data[col].fillna('').astype(str).map(len).value_counts().to_dict()).map(np.log1p)
        n=fea_len//2 if fea_len <=10 else 10 if fea_len<=40 else 20
        tsvd=decomposition.TruncatedSVD(n_components=n,n_iter=10)
        tsvd_transform=tsvd.fit_transform(tfidf_transform)
        df=pd.DataFrame(tsvd_transform,columns=[col+'_tfidf'+str(i) for i in range(1,n+1)])
        df.index=data.index.values
    data=data.join(df,how='left')
    data[col+'_descb_vc']=data[col].fillna('').astype(str).map(data[col].fillna('').astype(str).value_counts().to_dict()).map(np.log1p)
    return data.drop(col,axis=1)

def convert2float(x):
    try:
        return float(x)
    except:
        return None

def generate_train_data(data):
    train=pd.read_csv(os.path.join("input","meinian_round1_train_20180408.csv"),index_col=0,encoding='gbk')
    test=pd.read_csv(os.path.join("input","[new] meinian_round1_test_a_20180409.csv"),index_col=0,encoding='gbk')
    D={'收缩压':'y1','舒张压':'y2','血清甘油三酯':'y3','血清高密度脂蛋白':'y4','血清低密度脂蛋白':'y5'}
    train=data.join(train,how='inner')
    test=data.join(test,how='inner')
    labels=train.columns.tolist()[-5:]
    for label in labels:#log(y+1)
        if label=='血清甘油三酯':
            train[label]=train[label].apply(lambda x:x.replace('+','').replace('轻度乳糜','').replace('>','').replace('2.2.8','2.3').strip()).astype(float)
        train[label]=train[label].map(convert2float)
    train=train.loc[train.loc[:,labels].isnull().sum(axis=1)==0]
    s=(train.loc[:,labels]<0).sum(axis=1)
    train=train.loc[s[s==0].index.tolist()]
    test=test.loc[:,train.columns.tolist()]
    train.rename(columns=D,inplace=True)
    test.rename(columns=D,inplace=True)
    train.to_csv(os.path.join(outdir,'train.csv'),encoding='utf-8')
    test.to_csv(os.path.join(outdir,'test.csv'),encoding='utf-8')
    
if __name__=="__main__":
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    with open(r'F:\ProgrammingLanguages\python\Learning\wordcloud\stopwords.txt','r',encoding='utf-8') as f:
        stopwords = f.read().split('\n')
    if not os.path.exists(os.path.join(indir,'features.csv')):
        data=pd.read_csv(os.path.join(indir,"feature_042609.csv"))
        data.drop(['Systolic','Diastolic','triglyceride','HDL','LDL'],axis=1,inplace=True)
        data.set_index('vid',inplace=True)
        data['NumNan']=data.isnull().sum(axis=1)
        cols=[col for col in data.columns.tolist() if not 'cnt' in col]
        data=data.loc[:,cols]
        
        cat_cols,num_cols=get_col_type(data)
        for col in cat_cols:
            print(col)
            data=text2vector(data,col)
            print(data.shape)
        for col in num_cols:
            print(col)
            data[col]=data[col].apply(lambda x:np.log1p(x) if not pd.isnull(x) else None)
        cols=[col.replace('<','less_than').replace('，',' ').replace(',',' ').replace('。',' ') for col in data.columns.values]
        data.columns=cols
        data.to_csv(os.path.join(indir,'features.csv'),encoding='utf-8')
    else:
        data=pd.read_csv(os.path.join(indir,'features.csv'),index_col=0)
    
    generate_train_data(data)