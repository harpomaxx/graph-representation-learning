#!/bin/bash

PATH_REPO=`git rev-parse --show-toplevel` # show the path of the top-level directory

cd ${PATH_REPO}/rawdata/ctu-13/ip_bytes/ncol_sIP-dIP-BnoZeros/features/

PATH_PWD=`pwd`
mkdir clara_eucl_SINnormalizar clara_manh_SINnormalizar

Rscript ${PATH_REPO}/code/R/012_clara_SINnormalizar.R "euclidean" |& tee clara_eucl_SINnormalizar/salida_eucl_SINnormalizar.txt &
Rscript ${PATH_REPO}/code/R/012_clara_SINnormalizar.R "manhattan" |& tee clara_manh_SINnormalizar/salida_manh_SINnormalizar.txt &

