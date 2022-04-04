#!/bin/bash

PATH_REPO=`git rev-parse --show-toplevel` # show the path of the top-level directory

cd ${PATH_REPO}/rawdata/ctu-13/ip_bytes/ncol_sIP-dIP-B/ # Note: to run in /rawdata/ctu-13/ip_bytes/ncol_sIP-dIP-B

mkdir ../ncol_sIP-dIP-BnoZeros

for filename in ./*.ncol; do 
    base=`basename ${filename} .ncol`
    awk '{if ($3-ne0) print $1, $2, $3}' ${filename} | sed '1iorigin destination weight' > ${base}_noZeroB.ncol
done    

mv *_noZeroB.ncol ../ncol_sIP-dIP-BnoZeros/


