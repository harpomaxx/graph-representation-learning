#!/bin/bash

PATH_REPO=`git rev-parse --show-toplevel` # show the path of the top-level directory

cd ${PATH_REPO}/rawdata/ctu-13/ip_bytes/ncol_sIP-dIP-BnoZeros/ # Note: to run in /rawdata/ctu-13/ip_bytes/ncol_sIP-dIP-BnoZeros

PATH_PWD=`pwd`

mkdir -p features/LCC_undirected_weighted/time

for filename in ./*.ncol; do 
    base1=`basename ${filename} .ncol`
    base2=${base1%_*}
    python3 -u ${PATH_REPO}/code/python/007.1_LCC_undirected_weighted.py ${base2}  |& tee ${PATH_PWD}/features/LCC_undirected_weighted/time/time_${base2}.txt
done    

mv *.csv features/LCC_undirected_weighted/

