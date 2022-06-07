FROM rapidsai/rapidsai:latest

WORKDIR /usr/src/app

COPY BC_with_cg.py .
COPY capture20110815-2.binetflow.labels.gz.csv .

CMD ["python", "BC_with_cg.py"]
