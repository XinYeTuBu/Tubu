# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 10:15:57 2021

@author: ruj72813
"""
import pandas as pd

# 删除空白数据
data_pd = pd.read_excel('data.xlsx', sheet_name='data', header=0)
data_pd = data_pd.dropna()

# 查看相关系数，合并日期时间
data_cor = data_pd.corr()
data_pd['时间'] = data_pd['日期'].map(str) + ' ' + data_pd['时间'].map(str)

# 根据data_cor选取以下列
col = ['时间','涂布槽液位','基片定量','传动侧','波美度','打浆度','湿重','涂布率']
data = data_pd[col]

# 保存到csv中
data.to_csv('data.csv',index_label=False)

# 清除变量
del data_pd, data_cor, col, data
