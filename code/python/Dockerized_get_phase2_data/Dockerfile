FROM python:latest

WORKDIR /usr/src/app

RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install pandas
RUN pip install pycaret[full]

ENTRYPOINT ["python", "./code/python/Dockerized_kmeans/kmeans.py"]
