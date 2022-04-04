#!/bin/bash

PATH_REPO=`git rev-parse --show-toplevel`   # show the path of the top-level directory

cd ${PATH_REPO}/rawdata/ctu-13/ip_bytes/csv_sIP-dIP-dB-sB/  # Note: to run in /rawdata/ctu-13/ip_bytes/csv_sIP-dIP-dB-sB

mkdir ../ncol_sIP-dIP-B

for filename in ./*.csv; do 
    base=`basename ${filename} .csv`
    python3 ${PATH_REPO}/code/python/002_create_ncol.py ${base%_*_*} 
done    

mv *.ncol ../ncol_sIP-dIP-B/



