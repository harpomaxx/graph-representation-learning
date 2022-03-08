#!/bin/bash

# Note: to run in /rawdata/ctu-13 (directory that has compressed files)


mkdir -p ip_bytes/csvFiles

for filename in ./*.gz; do 
    gunzip $filename    # unzip
    grep "\"" ${filename%.*}    # check that there are no quotation marks (because there could be commas inside quotation marks)

    if [ $? -ne 0 ]; then
        awk -F "," '{if ($3=="tcp" || $3=="udp") print $4, $7, $13, $14}' ${filename%.*} > ${filename%.*.*.*}_tmp   # extract SrcAddr DstAddr TotBytes SrcBytes
        awk -v OFS="," '$3-=$4' ${filename%.*.*.*}_tmp | sed '1iSrcAddr,DstAddr,DstBytes,SrcBytes'> ${filename%.*.*.*}_ip_bytes.csv   # subtract TotBytes-SrcBytes (to obtain DstBytes)
                                                                                                                                      # create a .csv file
                                                                                                                                      # update header 
    else
        echo "Please use another method because there may be commas inside quotation marks" > ${filename%.*.*.*}_warning.txt
    fi
done    

mv *_ip_bytes.csv ip_bytes/csvFiles/
rm -r *_tmp

for filename in ./*.labels; do 
    zip -q ${filename#./*}.gz ${filename#./*}
done    

rm -r *.labels

