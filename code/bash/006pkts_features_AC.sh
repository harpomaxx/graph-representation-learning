#!/bin/bash

PATH_REPO=`git rev-parse --show-toplevel` # show the path of the top-level directory

cd ${PATH_REPO}/rawdata/ctu-13-2format/ncol_sIP-dIP-pkts_noZeros

PATH_PWD=`pwd`

mkdir -p features/AC/time

for filename in ./*.ncol; do 
    base1=`basename ${filename} .ncol`
    base2=${base1%_*}
    Rscript ${PATH_REPO}/code/R/006pkts_features_AC.R ${base2} > ${PATH_PWD}/features/AC/time/time_${base2}.txt
done    

mv *.csv features/AC/


