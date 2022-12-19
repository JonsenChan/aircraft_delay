# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:31:28 2022

@author: User
"""

import pandas as pd # 資料處理
import pyarrow as py 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# 匯入資料
data = pd.read_parquet("airline_weather_concat_1025")

dcorr = data.corr(method='pearson')
plt.figure(figsize=(80, 60),dpi=100)
sns.heatmap(data=dcorr, vmax=0.3)


plt.title('Correlation of Features', y=1.05,size=15)

            


# 資料查看所有的狀態與缺失值
data_info = pd.DataFrame({'unicos':data.nunique(),
                          'missing_data':data.isna().mean()*100,
                          'dtype':data.dtypes})

# 26~60 剔除出發地與目的地頻率過低的地區 (以幾個大機場作為資料集)
dest_count = data['Dest'].value_counts()
dest_count = len(dest_count)

origin_count = data['Origin'].value_counts()
origin_count = len(origin_count)

total_row = len(data.index)

# 將出發與目的地，做頻率編碼
Dest_map = data['Dest'].value_counts().to_dict()
data['Dest'] = data['Dest'].map(Dest_map)

Origin_map = data['Origin'].value_counts().to_dict()
data['Origin'] = data['Origin'].map(Origin_map)

dest_avg = total_row/dest_count
origin_avg = total_row/origin_count

def pick_dest_airport(x):
    if x < dest_avg:
        return 0
    else:
        return 1


def pick_origin_airport(x):
    if x < origin_avg:
        return 0
    else:
        return 1

data['dest_freq'] = data['Dest'].apply(lambda x :pick_dest_airport(x))

data = data.drop(data[(data['dest_freq'] == 0)].index)

# data['origin_freq'] = data['Origin'].apply(lambda x :pick_origin_airport(x))

# data['total_freq'] = data['dest_freq'] + data['origin_freq']

# data = data[data['total_freq'] != 0]

# 把天氣延遲是0的都剃除掉
data = data[data['WeatherDelay'].notnull()]

# 將天氣延遲時間作為分類
def WeatherDelaygroup30(x):
    if 0 <= x < 15:
        return 0
    elif 15 <= x < 45 :
        return 1
    elif 45 <= x < 75: 
        return 2
    elif 75 <= x < 105:
        return 3
    elif 105 <= x < 135:
        return 4
    elif 135 <= x < 165:
        return 5
    else :
        return 6
# 利用上面的函數，去重新給一個分組號碼
data['WeatherDelayGroups30'] = data['WeatherDelay'].apply(lambda x :WeatherDelaygroup30(x))


# 因為要做分組預測，所以最後的預測值要平均分配 (不可以9成資料都是同一類，將所有資料作分配)
mask = data[data['WeatherDelayGroups30'] == 0]
mask = mask.sample(n=50000)
dataset = data[data['WeatherDelayGroups30']!=0]
data = pd.concat([dataset,mask],axis=0)

# 加入新的特徵
data['WTSUMx'] = data['WT01_x']+data['WT02_x']+data['WT03_x']+data['WT04_x']+data['WT05_x']+data['WT06_x']+data['WT07_x']+data['WT08_x']+data['WT09_x']+data['WT10_x']+data['WT11_x']+data['WT18_x']
data['WTSUMy'] = data['WT01_y']+data['WT02_y']+data['WT03_y']+data['WT04_y']+data['WT05_y']+data['WT06_y']+data['WT07_y']+data['WT08_y']+data['WT09_y']+data['WT10_y']+data['WT11_y']+data['WT18_y']


# IATA_Code_Marketing_Airline 做一般編碼
from sklearn.preprocessing import LabelEncoder
data['IATA_Code_Marketing_Airline'] = LabelEncoder().fit_transform(data['IATA_Code_Marketing_Airline'])

# 選擇要的特徵欄位
data.columns
col_select =['Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek',
       'IATA_Code_Marketing_Airline', 'Origin', 'Dest', 'CRSDepTime',
       'CRSArrTime', 'DistanceGroup',
       
       'AWND_x', 'PRCP_x', 'TMAX_x', 'TMIN_x', 'WSF2_x',
       'WSF5_x', 'SNOW_x', 'WT01_x', 'WT02_x', 'WT03_x', 'WT04_x', 'WT05_x',
       'WT06_x', 'WT07_x', 'WT08_x', 'WT09_x', 'WT10_x', 'WT11_x', 'WT18_x',
       
       'AWND_y', 'PRCP_y', 'TMAX_y', 'TMIN_y', 'WSF2_y', 
       'WSF5_y', 'SNOW_y','WT01_y', 'WT02_y', 'WT03_y', 'WT04_y', 'WT05_y', 
       'WT06_y', 'WT07_y','WT08_y', 'WT09_y', 'WT10_y', 'WT11_y', 'WT18_y',
       
       'WTSUMx', 'WTSUMy','WeatherDelayGroups30']

# 將要的特徵產生一個新表格
dataset = data[col_select]

# 查看欄位之間的關係係數
dataset_corr = dataset.corr()


target = 'WeatherDelayGroups30'
y = dataset[target]
# X = dataset.loc[:, dataset.columns != 'ArrivalDelayGroups'] #select all columns but not the labels
X = dataset.loc[:, ~dataset.columns.isin([target])]


#### NORMALIZE X ####

# Normalize so everything is on the same scale. 
from sklearn import preprocessing
cols = X.columns
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
np_scaled = min_max_scaler.fit_transform(X)

# new data frame with the new scaled data. 
X = pd.DataFrame(np_scaled, columns = cols)


from sklearn.model_selection import GridSearchCV
import time
from xgboost import XGBClassifier

# 資料集做切割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y)

# 開始運算時間
start = time.perf_counter()

# xgb 做超參數
xgb = XGBClassifier()
params = {'objective':['binary:logistic'],    # 輸出概率
              'learning_rate': [0.01],    # 更新过程中用到的收缩步长 (0-1)
              'max_depth': [20],    # 树的最大深度 (1-無限)
              'min_child_weight': [10],    # 决定最小叶子节点样本权重和，加权和低于这个值时，就不再分裂产生新的叶子节点(0-無限)
              'subsample': [1],    # 这个参数控制对于每棵树，随机采样的比例 (0-1)
              'colsample_bytree': [0.6],    # 用来控制每颗树随机采样的列数的占比 (0-1)
              'n_estimators': [1000]    # n_estimators：弱學習器的数量 (0-無限)
              }    # 給定種子數，固定42

grid_xgb = GridSearchCV(estimator = xgb,
                        param_grid = params,
                        scoring = 'accuracy', `
                        cv = 5,
                        n_jobs = -1)

grid_xgb.fit(X_train, y_train)



y_pred = grid_xgb.predict(X_test)


print(classification_report(y_test,y_pred))


matrix = confusion_matrix(y_test,y_pred)
class_names = [i for i in range(len(y.value_counts().index))]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


print(grid_xgb.best_params_)
print(grid_xgb.score(X_test, y_test))

print("This time is being calculated")

end = time.perf_counter()

print(end - start)


# 將模型儲存起來
import os
import pickle
import gzip
# 儲存model
modeldir = './model'
if not os.path.isdir(modeldir):
        os.mkdir(modeldir)
        
with open(f'{modeldir}/grid.pickle', 'wb') as f:
    pickle.dump(grid_xgb, f)
with gzip.GzipFile(f'{modeldir}/grid.pgz', 'w') as f:
    pickle.dump(grid_xgb, f)





# 儲存模型與scaler
import joblib
joblib.dump(min_max_scaler,'./min_max_scaler.save')
# Load it
#scaler = joblib.load('scaler_cci30_5.save')