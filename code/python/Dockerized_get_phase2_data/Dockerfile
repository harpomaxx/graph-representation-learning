#FRom 3.8.13 python image
FROM python:3.8.13

WORKDIR /usr/src/app

RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install pandas
RUN pip install pycaret==2.3.10

ENTRYPOINT ["python", "./code/python/Dockerized_get_phase2_data/get_phase2_data.py"]

#I'm connected via ssh to a linux machine
#Now I want to send some files to the linux machine
#I'm using the following command
#scp -r /home/eduardo/Desktop/LABSIN/graph-representation-learning/data/csv/captures_features/ epavez@200.12.137.13:/../../home/epavez/graph-representation-learning/data/csv/captures_features/
