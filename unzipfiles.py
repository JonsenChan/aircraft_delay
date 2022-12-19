# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 15:29:40 2022

@author: User
"""

import zipfile

for i in range(2018,2023):
    for j in range(1,13):
        with zipfile.ZipFile('C:\\Users\\User\\Downloads\\On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_'+ str(i)+'_'+ str(j) +'.zip', 'r') as zf:
            # 列出每個檔案
            for name in zf.namelist():
                # 解壓縮指定檔案至 /my/folder 目錄
                zf.extract(name, path='C:\\Users\\User\\Downloads\\airline_data')