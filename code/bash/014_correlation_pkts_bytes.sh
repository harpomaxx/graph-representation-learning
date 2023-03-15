#!/bin/bash

PATH_REPO=`git rev-parse --show-toplevel`   # show the path of the top-level directory

cd ${PATH_REPO}/rawdata/ctu-13-2format/

mkdir pkts_bytes pkts_bytes_noZeros

for filename in ./*.2format; do 
    base=`basename ${filename} .binetflow.2format`
    awk -F "," -v OFS="," '{if ($3=="tcp" || $3=="udp") print $1, $2, $20,  $21, $22, $23}' ${filename} | sed '1iSrcAddr,DstAddr,SrcPkts,DstPkts,SrcBytes,DstBytes' > ${base}_pkts_bytes.csv
    awk -F "," -v OFS="," '{if (($3-ne0) && ($4-ne0) && ($5-ne0) && ($6-ne0)) print $0}' ${base}_pkts_bytes.csv | sed '1iSrcAddr,DstAddr,SrcPkts,DstPkts,SrcBytes,DstBytes' > ${base}_pkts_bytes_noZeros.csv
done

mv *_pkts_bytes.csv pkts_bytes/
mv *_pkts_bytes_noZeros.csv pkts_bytes_noZeros/

cd pkts_bytes_noZeros/

mkdir correlation

for filename in ./*.csv; do 
    base=`basename ${filename} .csv`
    Rscript ${PATH_REPO}/code/R/014_correlation_pkts_bytes.R ${base%_*_*_*} |& tee correlation/correlation_${base}.txt
done

#for (i in colnames(featuresNorm)[c(-1,-2,-9)]) {
#    featuresNormGraficar$features <- append(featuresNormGraficar$features, rep(i,length(featuresNorm$node)); 
#    featuresNormGraficar$values <- append(featuresNormGraficar$values,featuresNorm$i) }

#pktsBytesNZ<-readr::read_csv("capture20110810_pkts_bytes_noZeros.csv",show_col_types=FALSE)
#shapiro.test(pktsBytesNZ$SrcPkts[1:5000])
#resDst <-cor.test(pktsBytesNZ$DstPkts, pktsBytesNZ$DstBytes,  method = "spearman")
#ggscatter(pktsBytesNZ[,c(3,5)], x="SrcPkts", y="SrcBytes", add="reg.line", add.params=list(color="blue",fill="lightgray"), conf.int=TRUE, cor.coef=TRUE, cor.method="spearman")
