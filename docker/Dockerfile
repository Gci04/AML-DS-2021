FROM python:3.8-slim-buster

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /
COPY collaborative-filtering /collaborative-filtering
COPY main.py .

CMD uvicorn main:app --host 0.0.0.0