import streamlit as st
import pandas as pd
import numpy as np
import joblib
from XGB_API2023 import *
from numpy import asarray

dfcsv1 = pd.read_csv('output_v88.csv', sep=';', nrows=100000)
dfcsv2 = dfcsv1.fillna(0)
pd.set_option('display.max_columns', None)
X = dfcsv2.drop(columns=['category'])
Y = dfcsv2['category']
result =  pd.concat([X, Y.reindex(X.index)], axis=1)

rr = (asarray([result]))
data = st.text(rr)

model=joblib.load('dd44.joblib',mmap_mode = 'r+' )
prediction = model.predict(data)[0][0]

