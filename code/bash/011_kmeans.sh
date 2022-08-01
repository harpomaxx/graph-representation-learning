#!/bin/bash

PATH_REPO=`git rev-parse --show-toplevel` # show the path of the top-level directory

cd ${PATH_REPO}/rawdata/ctu-13/ip_bytes/ncol_sIP-dIP-BnoZeros/features/features_normalized

PATH_PWD=`pwd`
mkdir kmeans 

Rscript ${PATH_REPO}/code/R/011_kmeans.R 

mv *.txt kmeans/
mv features_normalized_and_kmeans.csv kmeans/

