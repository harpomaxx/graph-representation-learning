FROM tiagopeixoto/graph-tool:latest

WORKDIR /usr/src/app

RUN yes | pacman -S python-pip
RUN pip install --upgrade pip
RUN pip install optparse-pretty
RUN pip install numpy
RUN pip install pandas

ENTRYPOINT ["python", "./code/python/Dockerized_features/extract_features.py", "-i"]
