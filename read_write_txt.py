# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 15:33:19 2022

@author: User
"""

# tail_number 是一個 list
with open('tail_number1.txt','w') as temp_file:
    for item in temp_file:
				# 對於 %s 不理解可以看下方的連結資訊
        temp_file.write("%s\n" % item)
        # https://www.delftstack.com/zh-tw/howto/python/python-string-formatting/