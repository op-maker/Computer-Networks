FROM python:3.8-slim

RUN apt-get update -y

WORKDIR /usr/src/backend

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

