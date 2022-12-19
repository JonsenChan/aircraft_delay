# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 14:40:48 2022

@author: User
"""

import pyspark.pandas as ps
import pyspark.sql.functions as fn
import os

import pandas as pd
from pyspark.sql import SparkSession

os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

import numpy as np
import sys

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer

import pyspark.sql.types as typ
from pyspark.ml import Pipeline
import pyspark.ml.evaluation as ev


labels = [('INFANT_ALIVE_AT_REPORT', typ.IntegerType()),
          ('BIRTH_PLACE', typ.StringType()),
          ('MOTHER_AGE_YEARS', typ.IntegerType()),
          ('FATHER_COMBINE_AGE', typ.IntegerType()),
          ('CIG_BEFORE', typ.IntegerType()),
          ('CIG_1_TRI', typ.IntegerType()),
          ('CIG_2_TRI', typ.IntegerType()),
          ('CIG_3_TRI', typ.IntegerType()),
          ('MOTHER_HEIGHT_IN', typ.IntegerType()),
          ('MOTHER_PRE_WEIGHT', typ.IntegerType()),
          ('MOTHER_DELIVERY_WEIGHT', typ.IntegerType()),
          ('MOTHER_WEIGHT_GAIN', typ.IntegerType()),
          ('DIABETES_PRE', typ.IntegerType()),
          ('DIABETES_GEST', typ.IntegerType()),
          ('HYP_TENS_PRE', typ.IntegerType()),
          ('HYP_TENS_GEST', typ.IntegerType()),
          ('PREV_BIRTH_PRETERM', typ.IntegerType())
          ]

# schema = typ.StructType([
#     typ.StructField(e[0], e[1], False) for e in labels
# ])

data = ps.read_parquet('sample', header=True,index_col='__index_level_0__')

data.columns

data.dtypes


data_train, data_test = data.randomSplit([0.7,0.3],seed=123)

model = Pipeline.fit(data_train)
test_model = model.transform(data_test)
