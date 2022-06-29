#Base Image
From python:3.8-slim

COPY . /usr/app/
EXPOSE 5000
WORKDIR /usr/app/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt 

CMD python3 app.py
