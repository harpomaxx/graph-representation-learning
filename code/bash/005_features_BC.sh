#!/bin/bash

PATH_REPO=`git rev-parse --show-toplevel` # show the path of the top-level directory

cd ${PATH_REPO}/rawdata/ctu-13/ip_bytes/ncol_sIP-dIP-BnoZeros/ # Note: to run in /rawdata/ctu-13/ip_bytes/ncol_sIP-dIP-BnoZeros

PATH_PWD=`pwd`

mkdir -p features/BC/time

python3 -u ${PATH_REPO}/code/python/005_features_BC.py capture20110810  |& tee ${PATH_PWD}/features/BC/time/time_capture20110810.txt &
python3 -u ${PATH_REPO}/code/python/005_features_BC.py capture20110811  |& tee ${PATH_PWD}/features/BC/time/time_capture20110811.txt &
python3 -u ${PATH_REPO}/code/python/005_features_BC.py capture20110812  |& tee ${PATH_PWD}/features/BC/time/time_capture20110812.txt &
python3 -u ${PATH_REPO}/code/python/005_features_BC.py capture20110815-2  |& tee ${PATH_PWD}/features/BC/time/time_capture20110815-2.txt &
python3 -u ${PATH_REPO}/code/python/005_features_BC.py capture20110815-3  |& tee ${PATH_PWD}/features/BC/time/time_capture20110815-3.txt &
python3 -u ${PATH_REPO}/code/python/005_features_BC.py capture20110815  |& tee ${PATH_PWD}/features/BC/time/time_capture20110815.txt &
python3 -u ${PATH_REPO}/code/python/005_features_BC.py capture20110816-2  |& tee ${PATH_PWD}/features/BC/time/time_capture20110816-2.txt &
python3 -u ${PATH_REPO}/code/python/005_features_BC.py capture20110816-3  |& tee ${PATH_PWD}/features/BC/time/time_capture20110816-3.txt &
python3 -u ${PATH_REPO}/code/python/005_features_BC.py capture20110816  |& tee ${PATH_PWD}/features/BC/time/time_capture20110816.txt &
python3 -u ${PATH_REPO}/code/python/005_features_BC.py capture20110817  |& tee ${PATH_PWD}/features/BC/time/time_capture20110817.txt &
python3 -u ${PATH_REPO}/code/python/005_features_BC.py capture20110818-2  |& tee ${PATH_PWD}/features/BC/time/time_capture20110818-2.txt &
python3 -u ${PATH_REPO}/code/python/005_features_BC.py capture20110818  |& tee ${PATH_PWD}/features/BC/time/time_capture20110818.txt &
python3 -u ${PATH_REPO}/code/python/005_features_BC.py capture20110819  |& tee ${PATH_PWD}/features/BC/time/time_capture20110819.txt &

#mv *.csv features_part2/

