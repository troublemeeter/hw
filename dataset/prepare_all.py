import csv
import numpy as np
import os
import pandas as pd
import pickle

def prepare_train():

    if os.path.exists('dataset/train.data'):
        print('reading data from dataset/train.data ...')
        with open('dataset/train.data','rb') as f:
            train_data = pickle.load(f)
    train_data = train_data.dropna()
    y = train_data['age_group'].values - 1
    del train_data['uId']
    del train_data['age_group']
    x = train_data.values

    print('x shape: ',x.shape,' y shape: ',y.shape)

    return x,y

def prepare_test():

    if os.path.exists('dataset/test.data'):
        print('reading data from dataset/test.data ...')
        with open('dataset/test.data','rb') as f:
            test_data = pickle.load(f)

    test_data = test_data.dropna()

    del test_data['uId']
    x = test_data.values

    print('x shape: ',x.shape)

    return x