
# 1. Library imports
from typing_extensions import Self
import uvicorn
from fastapi import FastAPI
from XGBAPI_2023 import series_to_supervised, train, train_test_split, walk_forward_validation, xgboost_forecast, convert

from pathlib import Path
#from RFAPI_2023 import  xgboost_forecast, walk_forward_validation
from fastapi.responses import HTMLResponse
from typing import Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from numpy import asarray
import joblib

from typing import Annotated

from fastapi import FastAPI, Query

app = FastAPI()

class StockIn(BaseModel):
    X: list[int] = []
    testX:list[int] =[]
    data:list[int] =[]
    train:list[int] =[]
    datauser:list[int] = []
    
class StockIn1(BaseModel):
    
    datauser:list[int] = []

class StockOut(StockIn):

    yhat: int
    data_predict: list[int] =[]

@app.get("/")
def root(testX:StockIn):
    model = joblib.load('dd.joblib',mmap_mode = 'r+' )
    yhat = model.predict(asarray([testX]))
    json_data = jsonable_encoder(yhat)
    return JSONResponse(content=json_data)

#http://127.0.0.1:8000/items/5?q=somequery

# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
#if __name__ == '__main__':
#    uvicorn.run(app, host='127.0.0.1', port=8002)

@app.post("/predict")
def predict(data: StockIn):
    prediction =  walk_forward_validation(data,2)

    return {"prediction": prediction}
    
@app.get("/preduction", response_model=StockOut, status_code=200)
def get_prediction(datauser: StockIn1):

    yhat = walk_forward_validation(datauser, 24)
    json_yhat = jsonable_encoder(yhat)
    return {"forecast": json_yhat}



@app.get("/predict_user", response_model=StockOut, status_code=200)
async def pong(train: StockIn, X: StockIn, data: StockIn,testX:StockIn):

    data_predict = []
    data_predict =  xgboost_forecast(train, X)

    yhat = walk_forward_validation(data, 24,testX)
    json_yhat = jsonable_encoder(yhat)
    return {"load": data, "forecast": json_yhat, "predict":  data_predict }


@app.post("/predict_2", response_model=StockOut, status_code=200)
def get_prediction(train: StockIn, X: StockIn, data: StockIn,testX:StockIn):
    
    data_predict =  xgboost_forecast(train, X)

    yhat = walk_forward_validation(data, 24,testX)
    json_yhat = jsonable_encoder(yhat)
    if not yhat:
        raise HTTPException(status_code=400, detail="Model not found.")

    response_object = {"load": data, "forecast": convert(yhat), "predict":  json_yhat, "dpredict": data_predict}
    return response_object



  
if __name__ == '__main__':
  uvicorn.run(app,host="127.0.0.1",port=8000)

#http://127.0.0.1:8000/docs#/default/get_prediction_predict_post

