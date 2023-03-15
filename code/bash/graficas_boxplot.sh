#!/bin/bash

PATH_REPO=`git rev-parse --show-toplevel`   # show the path of the top-level directory

cd ${PATH_REPO}/rawdata/ctu-13/ip_bytes/ncol_sIP-dIP-BnoZeros/features/features_normalized/

mkdir -p graficas/boxplot

for filename in ./*_features_normalized.csv; do 
    base=`basename ${filename} .csv`
    Rscript ${PATH_REPO}/code/R/graficas_boxplot.R ${base%_*_*}  
done    

mv *.png graficas/boxplot/
mv *.pdf graficas/boxplot/

