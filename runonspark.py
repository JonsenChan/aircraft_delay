# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 19:22:30 2022

@author: User
"""

import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
import numpy as np
import pandas as pd
import pyspark
import sys

import pyspark.pandas as ps
import pyspark.sql.functions as fn
from pyspark.sql import SparkSession

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
 
spark = SparkSession\
        .builder\
        .appName("iris")\
        .getOrCreate()


spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", True)
ps.set_option("compute.default_index_type","distributed")


import pyarrow as py
from sklearn.model_selection import train_test_split
import time
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRFClassifier
from xgboost import plot_tree, plot_importance

X = ps.read_parquet('X')
y = ps.read_parquet('y')

y = y.squeeze()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

start = time.perf_counter()

xgb = XGBClassifier()
params = {'objective':['binary:logistic'],    # 輸出概率
            'learning_rate': [0.3,0.1],    # 更新过程中用到的收缩步长 (0-1)
            'max_depth': [6,24],    # 树的最大深度 (1-無限)
            'min_child_weight': [1,10],    # 决定最小叶子节点样本权重和，加权和低于这个值时，就不再分裂产生新的叶子节点(0-無限)
            'subsample': [0.6,0.8],    # 这个参数控制对于每棵树，随机采样的比例 (0-1)
            'colsample_bytree': [0.6,0.8],    # 用来控制每颗树随机采样的列数的占比 (0-1)
            'n_estimators': [10,100],    # n_estimators：弱學習器的数量 (0-無限)
            'seed': [42]}    # 給定種子數，固定42

grid_xgb = GridSearchCV(estimator = xgb,
                        param_grid = params,
                        scoring = 'accuracy', 
                        cv = 5,
                        n_jobs = -1)

grid_xgb.fit(X_train, y_train)

y_pred = grid_xgb.predict(X_test)

print(grid_xgb.best_params_)
print(grid_xgb.score(X_test, y_test))

print("This time is being calculated")

end = time.perf_counter()

print(end - start)

