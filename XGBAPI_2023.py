# forecast monthly births with xgboost
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import AdaBoostRegressor
from matplotlib import pyplot
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from matplotlib import pyplot

from sklearn.datasets import make_classification
import pickle
import numpy as np
import pandas as pd

import schedule
import time
from pathlib import Path

from h3 import h3

from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
 
import uvicorn
from fastapi import FastAPI
import requests 

app = FastAPI()

dfcsv1 = pd.read_csv('output_v8888.csv', sep=';', nrows=10000)
dfcsv2 = dfcsv1.fillna(0)
pd.set_option('display.max_columns', None)
dfcsv4 = dfcsv2
X = dfcsv4.drop(columns=['category'])
Y = dfcsv2['category']
result =  pd.concat([X, Y.reindex(X.index)], axis=1)
#result.to_csv("output_v8888.csv")

# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	#n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values
 
# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]


def train(train):
	train = asarray(train)
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = OneVsRestClassifier(XGBClassifier()) #RandomForestClassifier(n_estimators=100,max_depth=20, random_state=0)#XGBRegressor(objective='reg:squarederror', n_estimators=1000,tree_method='gpu_hist')
	model.fit(trainX, trainy)
	joblib.dump(model,'dd.joblib', protocol=None, cache_size=None)
	return model,train
 
# fit an xgboost model and make a one step prediction
def xgboost_forecast(train, testX):
	# transform list into array
	#train = asarray(train)
	# split into input and output columns
	#trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	#model = RandomForestClassifier(n_estimators=100,max_depth=20, random_state=0)#XGBRegressor(objective='reg:squarederror', n_estimators=1000,tree_method='gpu_hist')
	#model.fit(trainX, trainy)
	# make a one-step prediction
	#joblib.dump(model, 'dd.joblib', compress=0, protocol=None, cache_size=None)
	#joblib.dump(model, Path(BASE_DIR)).joinpath(f"{dfcsv2}.joblib")
	#model_file = Path(BASE_DIR).joinpath(f"{dfcsv2}.joblib")

	model = joblib.load('dd.joblib',mmap_mode = 'readwrite')

	yhat = model.predict(asarray([testX]))

	## filename = 'finalized_model.sav'
	## pickle.dump(model, open(filename, 'wb'))
    
	# loaded_model = pickle.load(open(filename, 'rb'))
	# result1 = loaded_model.score(X,Y)
	
	return yhat[0] #,result1
 
# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
	
	#model = joblib.load('dd.joblib',mmap_mode = 'r+' )
	#yhat = model.predict(asarray([testX]))

	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat = xgboost_forecast(history, testX)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# estimate prediction error
	error = mean_absolute_error(test[:, -1], predictions)
	
	return error, test[:, -1], predictions

dataset = result#pd.read_csv('table_orders.csv')
#dataset = dataset.set_index('time')        
cols = list(dataset)
    #series = dataset[cols] 
# split dataset
V = dataset.values     
data = series_to_supervised(V, n_in=1)
train(data[0:8000]) # - переключение!!!!!!!!!!!!!!!!
mae, y, yhat = walk_forward_validation(data[8000:10000], 48)

def convert(yhat):
    output = {"prediction":yhat}
    
    return output

#response = requests.post('http://127.0.0.1:8008/predict', json=convert(yhat))
#print(response.content)

@app.post('/main/')
async def predict():
  #if request.method == 'GET':
   
    data = xgboost_forecast(result, X[0:1])
    
    
    return {"data": data}
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

print(result)
print('MAE: %.3f' % mae)

pyplot.plot(y, label='Expected')
pyplot.plot(yhat, label='Predicted')
print('Column Number : ')
pyplot.legend()
pyplot.show()
