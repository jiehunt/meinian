import pandas as pd
import numpy as np
# import xgboost as xgb
import lightgbm as lgb
import glob
import gc
from sklearn.model_selection import train_test_split

def h_get_train_test_list():
   oof_files= glob.glob("oof/*")
   train_list = []
   test_list = []

   for f in oof_files:
       train_list.append(f)
       # oof_path = str(f).split('/')[0]
       # oof_file = str(f).split('/')[1]
       oof_path = str(f).split('\\')[0]
       oof_file = str(f).split('\\')[1]
       oof_test_pre = str(oof_file).split('oof')[0]
       test_file = str(oof_path) + '_test/'+str(oof_test_pre) + 'oof_test.csv'
       test_list.append(test_file)

   return train_list, test_list

def h_prepare_data_train(file_list):
    class_names = ['Systolic','Diastolic','triglyceride','HDL','LDL']
    class_names_oof = []
    for c in class_names:
        class_names_oof.append(c+'_oof')

    # df = pd.read_csv('input/train.csv')
    df = pd.DataFrame()
    for (n, f) in enumerate(file_list):
        one_file = pd.read_csv(f)
        one_file_n = one_file[class_names]
        n_class_name = []
        for c in class_names_oof:
            n_class_name.append(c+str(n))

        one_file_n.columns = n_class_name
        if n > 0:
            df = pd.concat([df, one_file_n], axis=1)
        else:
            df = one_file_n

    return df

def h_prepare_data_test(file_list):
    class_names = ['Systolic','Diastolic','triglyceride','HDL','LDL']
    class_names_oof = []
    for c in class_names:
        class_names_oof.append(c+'_oof')

    df = pd.DataFrame()
    for (n, f) in enumerate(file_list):
        one_file = pd.read_csv(f,header=None)
        one_file_n = one_file
        n_class_name = ['vid']
        for c in class_names_oof:
            n_class_name.append(c+str(n))
        one_file_n.columns = n_class_name
        one_file_n.drop(['vid'], axis=1, inplace = True)
        if n > 0:
            df = pd.concat([df, one_file_n], axis=1)
        else:
            df = one_file_n

    return df

def app_stack():

    class_names = ['Systolic','Diastolic','triglyceride','HDL','LDL']

    data=pd.read_csv('input/feature_042609.csv',low_memory=False)
    data = data.rename(columns={"收缩压": "Systolic", "舒张压": "Diastolic", "血清甘油三酯":"triglyceride", "血清高密度脂蛋白":"HDL", "血清低密度脂蛋白":"LDL"})

    test_lenth=9538
    test=data[-test_lenth:]
    train=data[:-test_lenth]
    test_vid = test['vid']
    test_vid = test_vid.reset_index(drop=True)
    print (test_vid.head())
    a=train[class_names].describe(percentiles=[.005,.05,.10,.20,.25,.5,.75,.95,.99,.995])
    for class_name in class_names:
        train = train[np.isfinite(train[class_name])]
        train = train[train[class_name] > a.iloc[4,:][class_name]]
        train = train[train[class_name] < a.iloc[-2,:][class_name]]
    y_train = train[class_names]
    del test, train, data
    gc.collect()

    print (len(y_train))
    class_names_oof = []
    for c in class_names:
        class_names_oof.append(c+'_oof')

    train_list, test_list =  h_get_train_test_list()
    num_file = len(train_list)

    train = h_prepare_data_train(train_list)
    # print (train.head())
    print (len(train))
    test = h_prepare_data_test(test_list)
    print (len(test))

    # stacker = LogisticRegression()
    # stacker = xgb.XGBRegressor()
    stacker = lgb.LGBMRegressor(n_estimators=1000, boosting_type="gbdt", learning_rate=0.1)

    # X_train, X_valid, Y_train, Y_valid = train_test_split(train_r, y_train, test_size = 0.1, random_state=982)
    # Fit and submit
    X_train = train
    Y_train = y_train
    scores = []
    sub = pd.DataFrame()
    for label in class_names:
        print(label)
        stacker.fit(X_train, Y_train[label])
        sub[label] = stacker.predict(test)

    print (num_file)
    out_file = 'submission_' + str(num_file) +'file.csv'
    sub = pd.concat([test_vid, sub], axis=1)
    sub.to_csv(out_file,index=False, header = False)
    return

app_stack()


