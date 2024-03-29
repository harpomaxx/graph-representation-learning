#!/bin/bash

PATH_REPO=`git rev-parse --show-toplevel` # show the path of the top-level directory

cd ${PATH_REPO}/rawdata/ctu-13/ip_bytes/ncol_sIP-dIP-BnoZeros/features/features_normalized

PATH_PWD=`pwd`
mkdir kmeans_HW_sinAC

Rscript ${PATH_REPO}/code/R/011_kmeans.R "Hartigan-Wong" >> kmeans_HW_sinAC/salida_HW_sinAC.txt

mv *Hartigan-Wong_sinAC.txt kmeans_HW_sinAC/
mv *Hartigan-Wong_sinAC.csv kmeans_HW_sinAC/

###########################################################
# PRUEBAS CON AC

#mkdir kmeans_HW_unix_gc kmeans_L_unix_gc

#Rscript ${PATH_REPO}/code/R/011_kmeans.R "Hartigan-Wong" >> salida_HW.txt &
#Rscript ${PATH_REPO}/code/R/011_kmeans.R "Lloyd" >> salida_L.txt &

#mv *Hartigan-Wong.txt kmeans_HG_unix_gc/
#mv *Hartigan-Wong.csv kmeans_HG_unix_gc/

# Agregue Lloyd
#mv *Lloyd.txt kmeans_L_unix_gc/
#mv *Lloyd.csv kmeans_L_unix_gc/

#########################
# Por defecto HW
##mv features_normalized_and_kmeans.csv kmeans/
##mv HOB_BOB_table.csv kmeans/
