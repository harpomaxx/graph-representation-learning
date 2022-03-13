#!/bin/Rscript
#  Convert a netflow file to a CSV files with 4 tuples  

suppressPackageStartupMessages(library(optparse))
suppressPackageStartupMessages(library(igraph))

source("code/R/convert_net2graph.R")

option_list <- list(
  make_option("--input", action="store", type="character", help = "Set the name of the input netflow file"),
  make_option("--output", action="store", type="character", default="netflow-igraph.ncol", help = "Set the name of the output  ncol file")
)
opt <- parse_args(OptionParser(option_list=option_list))

if (opt$input %>% is.null() || opt$output %>% is.null()){
  message("[] Parameters missing. Please use --help for look at available parameters.")
  quit()
}else{
  net_graph <- convert_net2graph(input_file = opt$input)
  ## Save 4 tuples files
  dir.create(dirname(opt$output), showWarnings = FALSE)
  write_graph(net_graph,file=opt$output,format="ncol")
}