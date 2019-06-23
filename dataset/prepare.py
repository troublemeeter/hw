import csv
import numpy as np
import os
import pandas as pd
import pickle

def prepare():

    if os.path.exists('dataset/dataset.pkl'):
        print('reading data from dataset/dataset.pkl ...')
        with open('dataset/dataset.pkl','rb') as f:
            x,y = pickle.load(f)
        return x,y


    age_train = pd.read_csv('dataset/age_train.csv',header=None)
    age_train.columns = ['uId','age_group']
    app_info = pd.read_csv('dataset/app_info.csv',header=None)
    app_info.columns = ['appId','category']
    user_basic_info = pd.read_csv('dataset/user_basic_info.csv',header=None)
    user_basic_info.columns = ['uId','gender','city','prodName',
                               'ramCapacity','ramLeftRation','romCapacity',
                               'romLeftRation','color','fontSize',
                               'ct','carrier','os']
    user_app_actived = pd.read_csv('dataset/user_app_actived.csv',header=None)
    user_app_actived.columns = ['uId','appId']
    user_behavior_info = pd.read_csv('dataset/user_behavior_info.csv',header=None)
    user_behavior_info.columns = ['uId','bootTimes','AFuncTimes','BFuncTimes',
                                  'CFuncTimes','DFuncTimes','EFuncTimes',
                                  'FFuncTimes','FFuncSum']


    user_basic_info = user_basic_info.dropna()
    app_info = app_info.dropna()
    age_train = age_train.dropna()
    user_behavior_info = user_behavior_info.dropna()
    user_app_actived = user_app_actived.dropna()

    app_info = pd.get_dummies(app_info,columns=['category'])
    del user_basic_info['city']
    user_basic_info = pd.get_dummies(user_basic_info,columns=['prodName','color','ct','carrier'])
    # user_basic_info = pd.get_dummies(user_basic_info,columns=['city','prodName','color','ct','carrier'])
    # todo: city 和 prodName 维度太高，估计比较稀疏，要把数目少的类别归为一个other类。
    # city这个特征作用不大，而且有300多维，可能可以删了


    df = user_behavior_info
    float_col = [each for each in df.columns if df[each].dtype=='float']
    normalize_col = float_col + ['bootTimes','FFuncSum']
    user_behavior_info[normalize_col] = df[normalize_col].apply(lambda x: (x - np.mean(x)) / np.std(x))

    df = user_basic_info
    float_col = [each for each in df.columns if df[each].dtype=='float']
    user_basic_info[float_col] = df[float_col].apply(lambda x: (x - np.mean(x)) / np.std(x))                              


    app_info_dict = {}
    for each in app_info.values:
        app = each[0]
        cat = each[1:]
        app_info_dict[app] = cat
        
    def f(x):
        x = x.split('#')
        r = []
        for each in x:
            if each in app_info_dict.keys():
                r.append(app_info_dict[each])
            else:
                r.append(np.array([0]*40))
        return sum(r)
        
    user_app_actived['app_total_categery'] = user_app_actived['appId'].apply(f)


    user_app_actived['app_numbers'] = user_app_actived['appId'].apply(lambda x:len(x.split('#')))
    user_app_actived['app_average_categery'] = user_app_actived['app_total_categery'].apply(lambda x:x/sum(x))
    # user_app_actived['topk_app'] = user_app_actived['app_total_categery'].apply(lambda x:x/sum(x))
    # todo: 需要看app的分布直方图，选取topK或者top m 到 n。

    temp = user_app_actived[['uId','app_average_categery']].values
    temp1 = []
    for i in  range(len(temp)):
        temp1.append(np.insert(temp[i][1],0,temp[i][0]))
    temp1 = np.array(temp1)
    temp1 = pd.DataFrame(temp1)
    temp1.columns = ['uId'] + ['categery_'+str(i) for i in range(40)]
    temp1['uId'] = temp1['uId'].astype('int64')
    temp1[['categery_'+str(i) for i in range(40)]] = temp1[['categery_'+str(i) for i in range(40)]].astype('float64')
     
    user_app_actived = pd.merge(user_app_actived,temp1, on=['uId'], how='inner')
    del user_app_actived['app_total_categery']
    del user_app_actived['app_average_categery']
    del user_app_actived['appId']

    join_data = pd.merge(age_train,user_behavior_info, on=['uId'], how='inner')
    join_data = pd.merge(join_data,user_basic_info,on=['uId'],how='inner')
    join_data = pd.merge(join_data,user_app_actived,on=['uId'],how='inner')

    print('prepare data done...')
    print(join_data.info(memory_usage='deep'))
    # print(len(join_data)/2010000)

    y = join_data['age_group'].values
    del join_data['uId']
    del join_data['age_group']
    x = join_data.values

    print('x shape: ',x.shape,' y shape: ',y.shape)


    with open('dataset/dataset.pkl','wb') as f:
        print('saving to dataset/dataset.pkl')
        pickle.dump([x,y],f)


    return x,y
