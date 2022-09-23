FROM  amancevice/pandas:latest

WORKDIR /usr/src/app

RUN pip install --upgrade pip
RUN pip install igraph
RUN pip install ipaddress
RUN pip install optparse-pretty
RUN pip install pandas

ENTRYPOINT ["python", "./code/python/Dockerized_cap2graphml/capture2graphml.py", "-i"]
