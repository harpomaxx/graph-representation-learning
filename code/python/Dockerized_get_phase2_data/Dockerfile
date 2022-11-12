#FRom 3.8.13 python image
FROM python:3.8.13

WORKDIR /usr/src/app

RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install pandas
RUN pip install pycaret==2.3.10

ENTRYPOINT ["python", "./code/python/Dockerized_get_phase2_data/get_phase2_data.py"]


