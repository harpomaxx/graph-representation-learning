#!/bin/bash

PATH_REPO=`git rev-parse --show-toplevel` # show the path of the top-level directory

cd ${PATH_REPO}/rawdata/ctu-13-2format/ncol_sIP-dIP-pkts_noZeros/features

PATH_PWD=`pwd`
mkdir kmeans_HW_SINnormalizar # kmeans_L_SINnormalizar

Rscript ${PATH_REPO}/code/R/011pkts_kmeans_SINnormalizar.R "Hartigan-Wong" |& tee kmeans_HW_SINnormalizar/salida_HW_SINnormalizar.txt &
#Rscript ${PATH_REPO}/code/R/011pkts_kmeans_SINnormalizar.R "Lloyd" |& tee kmeans_L_SINnormalizar/salida_L_SINnormalizar.txt &


#mv *Hartigan-Wong_SINnormalizar.txt kmeans_HW_SINnormalizar/
#mv *Hartigan-Wong_SINnormalizar.csv kmeans_HW_SINnormalizar/

# Agregue Lloyd
#mv *Lloyd_SINnormalizar.txt kmeans_L_SINnormalizar/
#mv *Lloyd_SINnormalizar.csv kmeans_L_SINnormalizar/

