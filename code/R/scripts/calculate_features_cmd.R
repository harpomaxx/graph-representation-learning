#!/bin/Rscript
#  calculate graph features  
suppressPackageStartupMessages(library(optparse))
suppressPackageStartupMessages(library(igraph))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
source("code/R/functions/calculate_features.R")

option_list <- list(
  make_option("--input", action="store", type="character", help = "Set the name of the input ncol file"),
  make_option("--output", action="store", type="character", default="netflow-features.csv", help = "Set the name of the output  csv file")
)
opt <- parse_args(OptionParser(option_list=option_list))

if (opt$input %>% is.null() || opt$output %>% is.null()){
  message("[] Parameters missing. Please use --help for look at available parameters.")
  quit()
}else{
  net_graph<-read_graph(opt$input, format='ncol', directed = TRUE)
  message("[R] Calculating features for ", opt$input)
  features_f <- calculate_features(net_graph)
  
  ## Save features
  dir.create(dirname(opt$output), showWarnings = FALSE, recursive = TRUE)
  write_csv(features_f
            ,file = opt$output)
  write(x = "",paste0("features.",Sys.getpid(),".end"))
  message("[R] File stored in ", opt$output)
}