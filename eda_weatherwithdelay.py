# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 15:45:31 2022

@author: User
"""

import pandas as pd # 資料處理
import pyarrow as py 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# data = pd.read_csv('sample.csv')

data = pd.read_parquet("airline_weather_concat_1025")
# a = airline_data.head(1000)



mask = data[data['ArrivalDelayGroups']==0]
mask = mask.sample(n=50000)
mask['ArrivalDelayGroups'].value_counts()

data_info = pd.DataFrame({'unicos':data.nunique(),
                          'missing_data':data.isna().mean()*100,
                          'dtype':data.dtypes})


airline_col = data.columns.tolist()
mask1 = ~data['ArrivalDelayGroups'].isnull()

# 利用 loc 去抓取符合表格內容
data = data.loc[mask1,airline_col]

data = data[data['WeatherDelay'].notnull()]

data.reset_index(drop=True,inplace=True)

data.groupby('ArrivalDelayGroups').size()
delay_df = data[['ArrivalDelayGroups','WeatherDelay','SecurityDelay','LateAircraftDelay','CarrierDelay','NASDelay']]
for x in range(len(delay_df)):
    weather = delay_df.at[x,'WeatherDelay']
    lateaircraft = delay_df.at[x,'LateAircraftDelay']
    security = delay_df.at[x,'SecurityDelay']
    carrier = delay_df.at[x,'CarrierDelay']
    nas = delay_df.at[x,'NASDelay']
    if ((weather < lateaircraft) or (weather < security) or (weather < carrier) or (weather < nas)):
        delay_df.at[x,'WeatherDelay'] = 0
        
data['WeatherDelay'] = delay_df[['WeatherDelay']]
data = data[data['WeatherDelay'] != 0]

data.reset_index(drop=True,inplace=True)

# 58~162 很笨的寫法，但我也是沒空想怎麼改
delay_df = data[['AWND_x', 'PRCP_x','TMAX_x', 'TMIN_x', 'WSF2_x', 'WSF5_x', 'SNOW_x',
                    'AWND_y', 'PRCP_y','TMAX_y', 'TMIN_y', 'WSF2_y', 'WSF5_y', 'SNOW_y',
                    'WT01_x', 'WT02_x', 'WT03_x', 'WT04_x', 'WT05_x',
                    'WT06_x', 'WT07_x', 'WT08_x', 'WT09_x', 'WT10_x', 'WT11_x', 'WT18_x',
                    'WT01_y', 'WT02_y', 'WT03_y', 'WT04_y', 'WT05_y', 'WT06_y', 'WT07_y',
                    'WT08_y', 'WT09_y', 'WT10_y', 'WT11_y', 'WT18_y']]
for x in range(len(delay_df)):
    AWND_x = delay_df.at[x,'AWND_x']
    PRCP_x = delay_df.at[x,'PRCP_x']
    TMAX_x = delay_df.at[x,'TMAX_x']
    TMIN_x = delay_df.at[x,'TMIN_x']
    WSF2_x = delay_df.at[x,'WSF2_x']
    WSF5_x = delay_df.at[x,'WSF5_x']
    SNOW_x = delay_df.at[x,'SNOW_x']
    AWND_y = delay_df.at[x,'AWND_y']
    PRCP_y = delay_df.at[x,'PRCP_y']
    TMAX_y = delay_df.at[x,'TMAX_y']
    TMIN_y = delay_df.at[x,'TMIN_y']
    WSF2_y = delay_df.at[x,'WSF2_y']
    WSF5_y = delay_df.at[x,'WSF5_y']
    SNOW_y = delay_df.at[x,'SNOW_y']
    WT01_x = delay_df.at[x,'WT01_x']
    WT02_x = delay_df.at[x,'WT02_x']
    WT03_x = delay_df.at[x,'WT03_x']
    WT04_x = delay_df.at[x,'WT04_x']
    WT05_x = delay_df.at[x,'WT05_x']
    WT06_x = delay_df.at[x,'WT06_x']
    WT07_x = delay_df.at[x,'WT07_x']
    WT08_x = delay_df.at[x,'WT08_x']
    WT09_x = delay_df.at[x,'WT09_x']
    WT10_x = delay_df.at[x,'WT10_x']
    WT11_x = delay_df.at[x,'WT11_x']
    WT18_x = delay_df.at[x,'WT18_x']
    WT01_y = delay_df.at[x,'WT01_y']
    WT02_y = delay_df.at[x,'WT02_y']
    WT03_y = delay_df.at[x,'WT03_y']
    WT04_y = delay_df.at[x,'WT04_y']
    WT05_y = delay_df.at[x,'WT05_y']
    WT06_y = delay_df.at[x,'WT06_y']
    WT07_y = delay_df.at[x,'WT07_y']
    WT08_y = delay_df.at[x,'WT08_y']
    WT09_y = delay_df.at[x,'WT09_y']
    WT10_y = delay_df.at[x,'WT10_y']
    WT11_y = delay_df.at[x,'WT11_y']
    WT18_y = delay_df.at[x,'WT18_y']
    if (AWND_x < AWND_y) :
        delay_df.at[x,'AWND_x'] = AWND_y
    if PRCP_x < PRCP_y:
        delay_df.at[x,'PRCP_x'] = PRCP_y
    if TMAX_x < TMAX_y:
        delay_df.at[x,'TMAX_x'] = TMAX_y
    if TMIN_x > TMIN_y:
        delay_df.at[x,'TMIN_x'] = TMIN_y
    if WSF2_x < WSF2_y:
        delay_df.at[x,'WSF2_x'] = WSF2_y
    if WSF5_x < WSF5_y:
        delay_df.at[x,'WSF5_x'] = WSF5_y
    if SNOW_x < SNOW_y:
        delay_df.at[x,'SNOW_x'] = SNOW_y
    if WT01_x < WT01_y:
        delay_df.at[x,'WT01_x'] = WT01_y
    if WT02_x < WT02_y:
        delay_df.at[x,'WT02_x'] = WT02_y
    if WT03_x < WT03_y:
        delay_df.at[x,'WT03_x'] = WT03_y
    if WT04_x < WT04_y:
        delay_df.at[x,'WT04_x'] = WT04_y
    if WT05_x < WT05_y:
        delay_df.at[x,'WT05_x'] = WT05_y
    if WT06_x < WT06_y:
        delay_df.at[x,'WT06_x'] = WT06_y
    if WT07_x < WT07_y:
        delay_df.at[x,'WT07_x'] = WT07_y
    if WT08_x < WT08_y:
        delay_df.at[x,'WT08_x'] = WT08_y
    if WT09_x < WT09_y:
        delay_df.at[x,'WT09_x'] = WT09_y
    if WT10_x < WT10_y:
        delay_df.at[x,'WT10_x'] = WT10_y
    if WT11_x < WT11_y:
        delay_df.at[x,'WT11_x'] = WT11_y
    if WT18_x < WT18_y:
        delay_df.at[x,'WT18_x'] = WT18_y



data['AWND_x'] = delay_df[['AWND_x']]
data['PRCP_x'] = delay_df[['PRCP_x']]
data['TMAX_x'] = delay_df[['TMAX_x']]
data['TMIN_x'] = delay_df[['TMIN_x']]
data['WSF2_x'] = delay_df[['WSF2_x']]
data['WSF5_x'] = delay_df[['WSF5_x']]
data['SNOW_x'] = delay_df[['SNOW_x']]
data['WT01_x'] = delay_df[['WT01_x']]
data['WT02_x'] = delay_df[['WT02_x']]
data['WT03_x'] = delay_df[['WT03_x']]
data['WT04_x'] = delay_df[['WT04_x']]
data['WT05_x'] = delay_df[['WT05_x']]
data['WT06_x'] = delay_df[['WT06_x']]
data['WT07_x'] = delay_df[['WT07_x']]
data['WT08_x'] = delay_df[['WT08_x']]
data['WT09_x'] = delay_df[['WT09_x']]
data['WT10_x'] = delay_df[['WT10_x']]
data['WT11_x'] = delay_df[['WT11_x']]
data['WT18_x'] = delay_df[['WT18_x']]

data['ArrivalDelayGroups'].value_counts()
            
data = pd.concat([data,mask],axis=0)


drop_list = ['Year','Quarter','FlightDate','Tail_Number','DayofMonth','LateAircraftDelay','CarrierDelay','NASDelay','PRCP_y','WeatherDelay','AWND_y','TMAX_y','WSF2_y','SecurityDelay','WSF5_y','SNOW_y','TMIN_y','WT01_y', 'WT02_y', 'WT03_y', 'WT04_y', 'WT05_y', 'WT06_y', 'WT07_y',
'WT08_y', 'WT09_y', 'WT10_y', 'WT11_y', 'WT18_y']

data = data.drop(columns=drop_list)

data.columns
# 想要顯現中文字不過下面的方法好像不太可行，好像是要在 google lab 用
# sns.set_style("whitegrid")
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# 這邊我把溫度轉成攝氏，對於我們來說比較好觀察
data['TMAX_x'] = (data['TMAX_x']-32)*5/9
data['TMIN_x'] = (data['TMIN_x']-32)*5/9

# 因為會出現小數點，所以我這邊做一些調整
def temp_change(x):
    x = str(x).split('.')[0]
    x = int(x)
    return x

data['TMAX_x'] = data['TMAX_x'].apply(lambda x :temp_change(x))
data['TMIN_x'] = data['TMIN_x'].apply(lambda x :temp_change(x))

train_group = data.groupby('ArrivalDelayGroups')

train_mean_y = train_group['AWND_x'].agg([lambda x:np.mean(x)]).rename(columns = {'<lambda>':'AWND_Different_Delay_Levels_MEAN'})
# train_mean_y1 = train_group['AWND_x'].agg('mean') #series
fig, ax = plt.subplots(figsize = (50,25))
sns.barplot([int(it) for it in train_mean_y.index], train_mean_y.values[:,0])
plt.xlabel('Delay_Levels',fontsize=50)
plt.ylabel('AWND_at_Different_Delay_Levels',fontsize=45)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title('AWND and Delay Relation',fontsize=50)
plt.show()



# =============================================================================



train_mean_y = train_group['PRCP_x'].agg([lambda x:np.mean(x)]).rename(columns = {'<lambda>':'PRCP_Different_Delay_Levels_MEAN'})

fig, ax = plt.subplots(figsize = (50,25))
sns.barplot([int(it) for it in train_mean_y.index], train_mean_y.values[:,0])
plt.xlabel('Delay_Levels',fontsize=50)
plt.ylabel('PRCP_at_Different_Delay_Levels',fontsize=45)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title('PRCP and Delay Relation',fontsize=50)
plt.show()

# =============================================================================


train_mean_y = train_group['WSF2_x'].agg([lambda x:np.mean(x)]).rename(columns = {'<lambda>':'WSF2_Different_Delay_Levels_MEAN'})

fig, ax = plt.subplots(figsize = (50,25))
sns.barplot([int(it) for it in train_mean_y.index], train_mean_y.values[:,0])
plt.xlabel('Delay_Levels',fontsize=50)
plt.ylabel('WSF2_at_Different_Delay_Levels',fontsize=45)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title('WSF2 and Delay Relation',fontsize=50)
plt.show()


train_mean_y = train_group['WSF5_x'].agg([lambda x:np.mean(x)]).rename(columns = {'<lambda>':'WSF5_Different_Delay_Levels_MEAN'})

fig, ax = plt.subplots(figsize = (50,25))
sns.barplot([int(it) for it in train_mean_y.index], train_mean_y.values[:,0])
plt.xlabel('Delay_Levels',fontsize=50)
plt.ylabel('WSF5_at_Different_Delay_Levels',fontsize=45)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title('WSF5 and Delay Relation',fontsize=50)
plt.show()


train_mean_y = train_group['SNOW_x'].agg([lambda x:np.mean(x)]).rename(columns = {'<lambda>':'SNOW_Different_Delay_Levels_MEAN'})

fig, ax = plt.subplots(figsize = (50,25))
sns.barplot([int(it) for it in train_mean_y.index], train_mean_y.values[:,0])
plt.xlabel('Delay_Levels',fontsize=50)
plt.ylabel('SNOW_at_Different_Delay_Levels',fontsize=45)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title('SNOW and Delay Relation',fontsize=50)
plt.show()


train_mean_y = train_group['TMAX_x'].agg([lambda x:np.mean(x)]).rename(columns = {'<lambda>':'TMAX_Different_Delay_Levels_MEAN'})

fig, ax = plt.subplots(figsize = (50,25))
sns.barplot([int(it) for it in train_mean_y.index], train_mean_y.values[:,0])
plt.xlabel('Delay_Levels',fontsize=50)
plt.ylabel('TMAX_at_Different_Delay_Levels',fontsize=45)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title('TMAX and Delay Relation',fontsize=50)
plt.show()


train_mean_y = train_group['TMIN_x'].agg([lambda x:np.mean(x)]).rename(columns = {'<lambda>':'TMIN_Different_Delay_Levels_MEAN'})

fig, ax = plt.subplots(figsize = (50,25))
sns.barplot([int(it) for it in train_mean_y.index], train_mean_y.values[:,0])
plt.xlabel('Delay_Levels',fontsize=50)
plt.ylabel('TMIN_at_Different_Delay_Levels',fontsize=45)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title('TMIN and Delay Relation',fontsize=50)
plt.show()


wt_0x = data['WT01_x'].value_counts().index
train_mean_y = train_group['WT01_x'].agg([lambda x:np.mean(x)])

fig, ax = plt.subplots(figsize = (25,15))
sns.barplot([int(it) for it in train_mean_y.index], train_mean_y.values[:,0])
plt.xlabel('Delay_Levels',fontsize=40)
plt.ylabel('WT01_at_Different_Delay_Levels',fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=25)
plt.title('WT01 and Delay Relation',fontsize=40)
plt.show()

wt_0x = data['WT02_x'].value_counts().index
train_mean_y = train_group['WT02_x'].agg([lambda x:np.mean(x)])

fig, ax = plt.subplots(figsize = (25,15))
sns.barplot([int(it) for it in train_mean_y.index], train_mean_y.values[:,0])
plt.xlabel('Delay_Levels',fontsize=40)
plt.ylabel('WT02_at_Different_Delay_Levels',fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=25)
plt.title('WT02 and Delay Relation',fontsize=40)
plt.show()

wt_0x = data['WT03_x'].value_counts().index
train_mean_y = train_group['WT03_x'].agg([lambda x:np.mean(x)])

fig, ax = plt.subplots(figsize = (25,15))
sns.barplot([int(it) for it in train_mean_y.index], train_mean_y.values[:,0])
plt.xlabel('Delay_Levels',fontsize=40)
plt.ylabel('WT03_at_Different_Delay_Levels',fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=25)
plt.title('WT03 and Delay Relation',fontsize=40)
plt.show()


wt_0x = data['WT04_x'].value_counts().index
train_mean_y = train_group['WT04_x'].agg([lambda x:np.mean(x)])

fig, ax = plt.subplots(figsize = (25,15))
sns.barplot([int(it) for it in train_mean_y.index], train_mean_y.values[:,0])
plt.xlabel('Delay_Levels',fontsize=40)
plt.ylabel('WT04_at_Different_Delay_Levels',fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=25)
plt.title('WT04 and Delay Relation',fontsize=40)
plt.show()

wt_0x = data['WT05_x'].value_counts().index
train_mean_y = train_group['WT05_x'].agg([lambda x:np.mean(x)])

fig, ax = plt.subplots(figsize = (25,15))
sns.barplot([int(it) for it in train_mean_y.index], train_mean_y.values[:,0])
plt.xlabel('Delay_Levels',fontsize=40)
plt.ylabel('WT05_at_Different_Delay_Levels',fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=25)
plt.title('WT05 and Delay Relation',fontsize=40)
plt.show()

wt_0x = data['WT06_x'].value_counts().index
train_mean_y = train_group['WT06_x'].agg([lambda x:np.mean(x)])

fig, ax = plt.subplots(figsize = (25,15))
sns.barplot([int(it) for it in train_mean_y.index], train_mean_y.values[:,0])
plt.xlabel('Delay_Levels',fontsize=40)
plt.ylabel('WT06_at_Different_Delay_Levels',fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=25)
plt.title('WT06 and Delay Relation',fontsize=40)
plt.show()

wt_0x = data['WT07_x'].value_counts().index
train_mean_y = train_group['WT07_x'].agg([lambda x:np.mean(x)])

fig, ax = plt.subplots(figsize = (25,15))
sns.barplot([int(it) for it in train_mean_y.index], train_mean_y.values[:,0])
plt.xlabel('Delay_Levels',fontsize=40)
plt.ylabel('WT07_at_Different_Delay_Levels',fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=25)
plt.title('WT07 and Delay Relation',fontsize=40)
plt.show()

wt_0x = data['WT08_x'].value_counts().index
train_mean_y = train_group['WT08_x'].agg([lambda x:np.mean(x)])

fig, ax = plt.subplots(figsize = (25,15))
sns.barplot([int(it) for it in train_mean_y.index], train_mean_y.values[:,0])
plt.xlabel('Delay_Levels',fontsize=40)
plt.ylabel('WT08_at_Different_Delay_Levels',fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=25)
plt.title('WT08 and Delay Relation',fontsize=40)
plt.show()

wt_0x = data['WT09_x'].value_counts().index
train_mean_y = train_group['WT09_x'].agg([lambda x:np.mean(x)])

fig, ax = plt.subplots(figsize = (25,15))
sns.barplot([int(it) for it in train_mean_y.index], train_mean_y.values[:,0])
plt.xlabel('Delay_Levels',fontsize=40)
plt.ylabel('WT09_at_Different_Delay_Levels',fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=25)
plt.title('WT09 and Delay Relation',fontsize=40)
plt.show()

wt_0x = data['WT10_x'].value_counts().index
train_mean_y = train_group['WT10_x'].agg([lambda x:np.mean(x)])

fig, ax = plt.subplots(figsize = (25,15))
sns.barplot([int(it) for it in train_mean_y.index], train_mean_y.values[:,0])
plt.xlabel('Delay_Levels',fontsize=40)
plt.ylabel('WT10_at_Different_Delay_Levels',fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=25)
plt.title('WT10 and Delay Relation',fontsize=40)
plt.show()


wt_0x = data['WT11_x'].value_counts().index
train_mean_y = train_group['WT11_x'].agg([lambda x:np.mean(x)])

fig, ax = plt.subplots(figsize = (25,15))
sns.barplot([int(it) for it in train_mean_y.index], train_mean_y.values[:,0])
plt.xlabel('Delay_Levels',fontsize=40)
plt.ylabel('WT11_at_Different_Delay_Levels',fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=25)
plt.title('WT11 and Delay Relation',fontsize=40)
plt.show()

wt_0x = data['WT18_x'].value_counts().index
train_mean_y = train_group['WT18_x'].agg([lambda x:np.mean(x)])

fig, ax = plt.subplots(figsize = (25,15))
sns.barplot([int(it) for it in train_mean_y.index], train_mean_y.values[:,0])
plt.xlabel('Delay_Levels',fontsize=40)
plt.ylabel('WT18_at_Different_Delay_Levels',fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=25)
plt.title('WT18 and Delay Relation',fontsize=40)
plt.show()



distance = data['WT18_x'].value_counts().index
train_mean_y = train_group['WT18_x'].agg([lambda x:np.mean(x)])

fig, ax = plt.subplots(figsize = (25,15))
sns.barplot([int(it) for it in train_mean_y.index], train_mean_y.values[:,0])
plt.xlabel('Delay_Levels',fontsize=40)
plt.ylabel('WT18_at_Different_Delay_Levels',fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=25)
plt.title('WT18 and Delay Relation',fontsize=40)
plt.show()

# data.to_parquet('viz1025ver2')