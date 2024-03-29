#!/bin/bash

PATH_REPO=`git rev-parse --show-toplevel` # show the path of the top-level directory

cd ${PATH_REPO}/rawdata/ctu-13-2format/ncol_sIP-dIP-pkts

mkdir ../ncol_sIP-dIP-pkts_noZeros

for filename in ./*.ncol; do 
    base=`basename ${filename} .ncol`
    awk '{if ($3-ne0) print $1, $2, $3}' ${filename} > ${base}_noZeroPkts.ncol
done    

mv *_noZeroPkts.ncol ../ncol_sIP-dIP-pkts_noZeros/


