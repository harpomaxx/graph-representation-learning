FROM python:latest

WORKDIR /usr/src/app

RUN pip install --upgrade pip
RUN pip install optparse-pretty
RUN pip install numpy
RUN pip install pandas

ENTRYPOINT ["python", "./code/python/Dockerized_hob_bob/hob_bob.py"]
