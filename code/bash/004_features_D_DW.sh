#!/bin/bash

PATH_REPO=`git rev-parse --show-toplevel` # show the path of the top-level directory

cd ${PATH_REPO}/rawdata/ctu-13/ip_bytes/ncol_sIP-dIP-BnoZeros/ # Note: to run in /rawdata/ctu-13/ip_bytes/ncol_sIP-dIP-BnoZeros

PATH_PWD=`pwd`

mkdir -p features/D_DW/time

for filename in ./*.ncol; do 
    base1=`basename ${filename} .ncol`
    base2=${base1%_*}
    python3 -u ${PATH_REPO}/code/python/004_features_D_DW.py ${base2}  |& tee ${PATH_PWD}/features/D_DW/time/time_${base2}.txt
done    

mv *.csv features/D_DW/


