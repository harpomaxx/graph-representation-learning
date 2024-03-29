#!/bin/bash

PATH_REPO=`git rev-parse --show-toplevel`   # show the path of the top-level directory

cd ${PATH_REPO}/rawdata/ctu-13-2format/pkts_bytes

mkdir ../ncol_sIP-dIP-pkts

for filename in ./*.csv; do 
    base=`basename ${filename} .csv`
    python3 ${PATH_REPO}/code/python/002pkts_create_ncol.py ${base%_*_*} 
done    

mv *.ncol ../ncol_sIP-dIP-pkts/



