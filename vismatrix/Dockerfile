FROM python:3.7.11

RUN pip3 install flask flask-restful flask-cors opencv-python 

COPY . /app 

WORKDIR /app

EXPOSE 8000

CMD ["python3", "web.py"]