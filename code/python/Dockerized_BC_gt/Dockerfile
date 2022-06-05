FROM tiagopeixoto/graph-tool:latest

WORKDIR /usr/src/app

COPY BC_with_gt.py .
COPY capture20110815-2.binetflow.labels.gz.graphml .

CMD ["python", "BC_with_gt.py"]
