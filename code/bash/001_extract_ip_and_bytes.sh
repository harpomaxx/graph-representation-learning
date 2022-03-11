#!/bin/bash

PATH_REPO=`git rev-parse --show-toplevel`   # show the path of the top-level directory

cd ${PATH_REPO}/rawdata/ctu-13/    # Note: to run in /rawdata/ctu-13 (directory that has compressed files)

mkdir -p ip_bytes/csv_sIP-dIP-dB-sB

for filename in ./*.gz; do 
    gunzip $filename    # unzip
    grep "\"" ${filename%.*}    # check that there are no quotation marks (because there could be commas inside quotation marks)

    if [ $? -ne 0 ]; then
        awk -F "," '{if ($3=="tcp" || $3=="udp") print $4, $7, $13, $14}' ${filename%.*} > ${filename%.*.*.*}_tmp   # filter protocols "tcp" and "udp"
                                                                                                                    # extract SrcAddr DstAddr TotBytes SrcBytes
                                                                                                                    
        awk -v OFS="," '{$3-=$4}{print $0}' ${filename%.*.*.*}_tmp | sed '1iSrcAddr,DstAddr,DstBytes,SrcBytes' > ${filename%.*.*.*}_ip_bytes.csv   # subtract TotBytes-SrcBytes (to obtain DstBytes)
                                                                                                                                                   # add header 
                                                                                                                                                   # create a .csv file
    else
        echo "Please use another method because there may be commas inside quotation marks" > ${filename%.*.*.*}_warning.txt
    fi
done    

mv *_ip_bytes.csv ip_bytes/csv_sIP-dIP-dB-sB/
rm -r *_tmp

for filename in ./*.labels; do 
    zip -q ${filename#./*}.gz ${filename#./*}
done    

rm -r *.labels


