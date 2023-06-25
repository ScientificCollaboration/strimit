# 

FROM python:3.9

# 

WORKDIR /app

# 

COPY ./requirements.txt /code/requirements.txt

# 

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
DOCKER LOGIN localhost:8080
#COPY ./app /app
ENTRYPOINT ["uvicorn", "app:app","--reload", "--host", "127.0.0.0", "--port", "8080"]]
# 
#ENTRYPOINT ["uvicorn", "app:app --reload"]
#CMD ["uvicorn", "app.app:app"," --reload", "--host", "0.0.0.0", "--port", "80"]
