# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 18:24:22 2022

@author: User
"""

import pandas as pd
import pyarrow as py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import pickle
import gzip
from pandas_ods_reader import read_ods
data_pred = read_ods('future_pred.ods')



# 將 index 整理好
data_pred.reset_index(drop=True,inplace=True)

# 使用者匯入資料的時候，並不會有 "WTSUMx","WTSUMy" 這兩個特徵欄位，所以在這邊加入新特徵
data_pred['WTSUMx'] = (data_pred['WT01_x']+data_pred['WT02_x']+data_pred['WT03_x']+data_pred['WT04_x']+
                       data_pred['WT05_x']+data_pred['WT06_x']+data_pred['WT07_x']+data_pred['WT08_x']+
                       data_pred['WT09_x']+data_pred['WT10_x']+data_pred['WT11_x']+data_pred['WT18_x'])

data_pred['WTSUMy'] = (data_pred['WT01_y']+data_pred['WT02_y']+data_pred['WT03_y']+data_pred['WT04_y']+
                       data_pred['WT05_y']+data_pred['WT06_y']+data_pred['WT07_y']+data_pred['WT08_y']+
                       data_pred['WT09_y']+data_pred['WT10_y']+data_pred['WT11_y']+data_pred['WT18_y'])


if data_pred['WTSUMx'][0]<5:
    df_pred = 1
    
df_pred = pd.DataFrame([df_pred])
