# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 15:29:04 2022

@author: User
"""

import numpy as np
import pandas as pd
import time as tm
a = pd.read_csv('tail_no_detail.csv')

# 針對'Status'欄位裡面符合'Renewal Date'、空值、'<td data-label="Status">Valid</td>'
# 用isin選取，再用~做反向選取
df = a[~a['Status'].isin(["Renewal Date",np.nan,
													'<td data-label="Status">Valid</td>'])]

# 針對'MFR Year'欄位裡面符合'0000'、空值、'None'
# 用isin選取，再用~做反向選取
df2 = df[~(df['MFR Year'].isin(["0000",np.nan,"None"]))]

# 產生不要欄位的list
colum_list=['Dealer','Type Certificate Data Sheet','Type Certificate Holder',
            'Engine Manufacturer', 'Classification','Engine Model', 'A/W Date',
            'Exception Code','Serial Number','Status','Expiration Date',
            'Fractional Owner','Pending Number Change','Mode S Code (base 8 / Oct)',
            'Mode S Code (Base 16 / Hex)','City','State','County','Zip Code',
            'Certificate Number','Issue Date','Expiration Date.1','Category',
            'Certificate Issue Date','Date Change Authorized','Type Registration']

# 針對上述不要的欄位，將欄位剔除
df3 = df2.drop(colum_list,axis=1)

# 抓現在時間，抓年份
curr = tm.localtime()
curr_year = curr.tm_year

# 將'MFR Year'欄位轉成int，創新欄位'Age'並給與程式現今年份扣掉製造年份
df3['MFR Year']= df3['MFR Year'].astype('int')
df3['Age']=(curr_year+1)-df3['MFR Year']
df3.to_csv('clean_tail2.csv')