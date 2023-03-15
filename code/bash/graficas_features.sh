#!/bin/bash

PATH_REPO=`git rev-parse --show-toplevel`   # show the path of the top-level directory

cd ${PATH_REPO}/rawdata/ctu-13/ip_bytes/ncol_sIP-dIP-BnoZeros/features/features_normalized

mkdir graficas

for filename in ./*_features_normalized.csv; do 
    base=`basename ${filename} .csv`
    lineas=`wc -l ${filename}`
    python3 ${PATH_REPO}/code/python/graficas_features.py ${base%_*_*} ${lineas} 
done    

mv *.png graficas/


