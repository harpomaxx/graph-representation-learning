#!/bin/Rscript

args <- commandArgs(TRUE)
csvName <- paste(args[1], "_pkts_bytes_noZeros.csv", sep = "")
cat("CAPTURA: ", args[1], "\n")


pktsBytesNZ <- readr::read_csv(csvName, show_col_types = FALSE)

#for (i in seq(3, length(pktsBytesNZ))) {
#    shapRes <- shapiro.test(pktsBytesNZ[1:5000,i])
#    cat("\nshapiro-wilk test (first 5000 rows): ", i, "\n")
#    print(shapRes)
#}
shapRes <- shapiro.test(pktsBytesNZ$SrcPkts[1:5000])
cat("\nSrcPkts -- shapiro-wilk test (first 5000 rows): \n")
print(shapRes)
shapRes <- shapiro.test(pktsBytesNZ$SrcBytes[1:5000])
cat("\nSrcBytes -- shapiro-wilk test (first 5000 rows): \n")
print(shapRes)
shapRes <- shapiro.test(pktsBytesNZ$DstPkts[1:5000])
cat("\nDstPkts -- shapiro-wilk test (first 5000 rows): \n")
print(shapRes)
shapRes <- shapiro.test(pktsBytesNZ$DstBytes[1:5000])
cat("\nDstBytes -- shapiro-wilk test (first 5000 rows): \n")
print(shapRes)


SrcCorrelation <- cor.test(pktsBytesNZ$SrcPkts, pktsBytesNZ$SrcBytes,  method = "spearman")
cat("SrcCorrelation \n")
print(SrcCorrelation)


DstCorrelation <-cor.test(pktsBytesNZ$DstPkts, pktsBytesNZ$DstBytes,  method = "spearman")
cat("DstCorrelation \n")
print(DstCorrelation)


print(warnings())
