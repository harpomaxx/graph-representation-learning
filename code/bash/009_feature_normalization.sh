#!/bin/bash

PATH_REPO=`git rev-parse --show-toplevel` # show the path of the top-level directory

cd ${PATH_REPO}/rawdata/ctu-13/ip_bytes/ncol_sIP-dIP-BnoZeros/ # Note: to run in /rawdata/ctu-13/ip_bytes/ncol_sIP-dIP-BnoZeros

PATH_PWD=`pwd`
DEPTH=1

mkdir -p features/features_normalized/time

Rscript ${PATH_REPO}/code/R/009_feature_normalization.R capture20110810 ${DEPTH} |& tee ${PATH_PWD}/features/features_normalized/time/time_capture20110810.txt &
Rscript ${PATH_REPO}/code/R/009_feature_normalization.R capture20110811 ${DEPTH} |& tee ${PATH_PWD}/features/features_normalized/time/time_capture20110811.txt &
Rscript ${PATH_REPO}/code/R/009_feature_normalization.R capture20110812 ${DEPTH} |& tee ${PATH_PWD}/features/features_normalized/time/time_capture20110812.txt &
Rscript ${PATH_REPO}/code/R/009_feature_normalization.R capture20110815-2 ${DEPTH} |& tee ${PATH_PWD}/features/features_normalized/time/time_capture20110815-2.txt &
Rscript ${PATH_REPO}/code/R/009_feature_normalization.R capture20110815-3 ${DEPTH} |& tee ${PATH_PWD}/features/features_normalized/time/time_capture20110815-3.txt &
Rscript ${PATH_REPO}/code/R/009_feature_normalization.R capture20110815 ${DEPTH} |& tee ${PATH_PWD}/features/features_normalized/time/time_capture20110815.txt &
Rscript ${PATH_REPO}/code/R/009_feature_normalization.R capture20110816-2 ${DEPTH} |& tee ${PATH_PWD}/features/features_normalized/time/time_capture20110816-2.txt &
Rscript ${PATH_REPO}/code/R/009_feature_normalization.R capture20110816-3 ${DEPTH} |& tee ${PATH_PWD}/features/features_normalized/time/time_capture20110816-3.txt &
Rscript ${PATH_REPO}/code/R/009_feature_normalization.R capture20110816 ${DEPTH} |& tee ${PATH_PWD}/features/features_normalized/time/time_capture20110816.txt &
Rscript ${PATH_REPO}/code/R/009_feature_normalization.R capture20110817 ${DEPTH} |& tee ${PATH_PWD}/features/features_normalized/time/time_capture20110817.txt &
Rscript ${PATH_REPO}/code/R/009_feature_normalization.R capture20110818-2 ${DEPTH} |& tee ${PATH_PWD}/features/features_normalized/time/time_capture20110818-2.txt &
Rscript ${PATH_REPO}/code/R/009_feature_normalization.R capture20110818 ${DEPTH} |& tee ${PATH_PWD}/features/features_normalized/time/time_capture20110818.txt &
Rscript ${PATH_REPO}/code/R/009_feature_normalization.R capture20110819 ${DEPTH} |& tee ${PATH_PWD}/features/features_normalized/time/time_capture20110819.txt &

#mv *normalized.csv features/features_normalized/

