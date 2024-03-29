#!/bin/bash

PATH_REPO=`git rev-parse --show-toplevel` # show the path of the top-level directory

cd ${PATH_REPO}/rawdata/ctu-13-2format/ncol_sIP-dIP-pkts_noZeros/features/features_normalized

PATH_PWD=`pwd`
mkdir kmeans_HW_normalizado

Rscript ${PATH_REPO}/code/R/011pkts_kmeans.R "Hartigan-Wong" |& tee kmeans_HW_normalizado/salida_HW_normalizado.txt &

#mv *Hartigan-Wong_normalizado.txt kmeans_HW_normalizado/
#mv *Hartigan-Wong_normalizado.csv kmeans_HW_normalizado/
