#!/bin/bash

PATH_REPO=`git rev-parse --show-toplevel` # show the path of the top-level directory

cd ${PATH_REPO}/rawdata/ctu-13/ip_bytes/ncol_sIP-dIP-BnoZeros/features/features_normalized

PATH_PWD=`pwd`
mkdir -p one-svm/usando_kmeans25

Rscript ${PATH_REPO}/code/R/012_one-svm.R >> one-svm/usando_kmeans25/salida25_one-svm.txt

mv capture20110817_predict25.csv one-svm/usando_kmeans25/
