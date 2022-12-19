# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 16:29:56 2022

@author: user
"""

import pandas as pd
import pyarrow as py


weather_data = pd.read_parquet('D:/Shared/專題資料/爬蟲/weather_data/weather_data_1006')
place = pd.read_excel('D:/Shared/專題資料/爬蟲/瑣碎資料/機場對照.xlsx')
airline_data = pd.read_parquet('C:/Users/User/Desktop/airport_data/airplane_delay_detect/airline_final_1019')
tail_number = pd.read_csv('D:/Shared/專題資料/爬蟲/Tail_No/clean_tail_withage.csv')



# 處理 place 的空值，全部 drop 掉
place.drop(columns=['IATA','Airport','Unnamed: 4'],inplace=True)
place.dropna(inplace=True)
place.rename(columns={'機場全稱': 'NAME', 
                      '機場簡稱': 'short_name'},inplace=True)


# 合併表格，需要相同的欄位去做對應，所以我下面將 日期、機場全稱和機場簡稱 合併成一個欄位，這樣相同的日期和地方就會對應

# 把日期格式轉換成 我想要的格式 2018/6/8 --> 2018-6-8
weather_data['DATE'] = weather_data["DATE"].str.replace('/','-')

# 日期格式須轉換為 6 --> 06 
# 先將 DATE 切割成三個部分 Year,Month,Day
weather_data[['Year','Month','Day']] = weather_data['DATE'].str.split(pat='-',expand=True)

# Month、Day 前面補一個0，e.g. 1 --> 01，12 --> 012
weather_data['Month'] = '0' + weather_data['Month']
weather_data['Day'] = '0' + weather_data['Day']

# 再取數字的後兩位，即可達成  01、02、03、...、10、11、12
weather_data['Month'] = weather_data['Month'].str[-2:]
weather_data['Day'] = weather_data['Day'].str[-2:]

# 將 Year,Month,Day 合併成 DATE，格式為 2018-06-08
weather_data['DATE'] = weather_data['Year'] +'-'+ weather_data['Month'] + '-' + weather_data['Day'] 

# 合併兩個檔案
weather_data = pd.merge(weather_data, place)


# 再加一個特徵欄位 DATE_AND_PLACE，為後面的資料表格做合併，內容為：日期 機場全名 機場簡稱
#                                                          2018-06-08 ALBANY SW GEORGIA REGIONAL AIRPORT, GA US ABY
weather_data['DATE_AND_PLACE'] = weather_data["DATE"] + ' ' + weather_data["NAME"] + ' ' + weather_data["short_name"]

# 把剛剛拆開來的資料，全部丟棄
weather_data.drop(columns=['Year','Month','Day'],inplace=True)


# 處理 airline_data 的內容
# airline_data 合併資料
airline_data = pd.merge(airline_data, place, left_on='Dest', right_on='short_name').drop("short_name",axis=1)
airline_data = pd.merge(airline_data, place, left_on='Origin', right_on='short_name').drop("short_name",axis=1)
a = airline_data.head()

# 再加一個特徵欄位 DATE_AND_PLACE，為後面的資料表格做合併，內容為：日期 機場全名 機場簡稱
#                                                          2018-06-08 ALBANY SW GEORGIA REGIONAL AIRPORT, GA US ABY
airline_data['DATE_AND_DestPLACE'] = airline_data["FlightDate"] + ' ' + airline_data["NAME_x"] + ' ' + airline_data["Dest"]

# 利用目的地的天氣
airline_data['DATE_AND_OriginPLACE'] = airline_data["FlightDate"] + ' ' + airline_data["NAME_y"] + ' ' + airline_data["Origin"]

drop_list = ['ArrDelay','ArrDelayMinutes','DepDelay',
             'DepDelayMinutes','DepDel15','DepTime','TaxiOut',
             'TaxiIn','ActualElapsedTime','ArrTime','Diverted',
             'DivAirportLandings','AirTime','CancellationCode','WheelsOn',
             'WheelsOff','ArrTimeBlk','DepTimeBlk','Distance','Cancelled']
airline_data = airline_data.drop(columns=drop_list)
# 一樣再次合併
airline_data = pd.merge(airline_data, weather_data, left_on='DATE_AND_DestPLACE', right_on='DATE_AND_PLACE')
airline_data = pd.merge(airline_data, weather_data, left_on='DATE_AND_OriginPLACE', right_on='DATE_AND_PLACE')
a = airline_data.head()

col_list = airline_data.columns.tolist()

# drop 掉不要的資料(記憶體不夠，所以一個一個刪除)
airline_data.drop(columns=['NAME_x'],inplace=True)
airline_data.drop(columns=['NAME_y'],inplace=True)
airline_data.drop(columns=['DATE_AND_DestPLACE'],inplace=True)
airline_data.drop(columns=['DATE_AND_OriginPLACE'],inplace=True)

airline_data.drop(columns=['STATION_x'],inplace=True)
airline_data.drop(columns=['DATE_x'],inplace=True)
airline_data.drop(columns=['short_name_x'],inplace=True)
airline_data.drop(columns=['DATE_AND_PLACE_x'],inplace=True)

airline_data.drop(columns=['STATION_y'],inplace=True)
airline_data.drop(columns=['DATE_y'],inplace=True)
airline_data.drop(columns=['short_name_y'],inplace=True)
airline_data.drop(columns=['DATE_AND_PLACE_y'],inplace=True)


# 移動機場全稱的欄位位置到簡稱的旁邊
airline_col = airline_data.columns.to_list()

airline_col.insert(airline_col.index('Dest')+1,
                    airline_col.pop(airline_col.index('NAME')))

airline_data = airline_data[airline_col]
airline_col = airline_data.columns.to_list()


# 確認機尾資料的欄位名稱叫什麼
t_col = tail_number.columns.tolist()

# 機尾資料的合併
airline_data = pd.merge(airline_data, tail_number, left_on='Tail_Number', right_on='Tail Number').drop("Tail_Number",axis=1)


airline_data = airline_data.drop(["Unnamed: 0"],axis=1)
airline_data = airline_data.drop(["Age"],axis=1)

# 最後存成一個檔
airline_data.to_parquet('airline_weather_concat_1026')
