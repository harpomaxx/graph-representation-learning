#!/bin/bash

PATH_REPO=`git rev-parse --show-toplevel` # show the path of the top-level directory

cd ${PATH_REPO}/rawdata/ctu-13/ip_bytes/ncol_sIP-dIP-BnoZeros/ # Note: to run in /rawdata/ctu-13/ip_bytes/ncol_sIP-dIP-BnoZeros

PATH_PWD=`pwd`
PATH_D_DW=${PATH_PWD}/features/D_DW/
PATH_BC=${PATH_PWD}/features/BC/
PATH_LCC=${PATH_PWD}/features/LCC_undirected_weighted/
PATH_AC=${PATH_PWD}/features/AC/

for filename in ./*.ncol; do 
    base1=`basename ${filename} .ncol`
    base2=${base1%_*}
    Rscript ${PATH_REPO}/code/R/008_features_all.R ${base2} ${PATH_D_DW} ${PATH_BC} ${PATH_LCC} ${PATH_AC} 
done    

mv *.csv features/
