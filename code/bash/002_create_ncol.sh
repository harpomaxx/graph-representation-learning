#!/bin/bash

# Note: to run in /rawdata/ctu-13/ip_bytes/csvFiles


mkdir ../ncolFiles

for filename in ./*.csv; do 
    base=`basename ${filename} .csv`
    python3 ../../../../code/python/002_create_ncol.py ${base%_*_*} 
done    

mv *.ncol ../ncolFiles/

