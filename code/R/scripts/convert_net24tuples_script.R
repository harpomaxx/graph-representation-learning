#!/bin/Rscript
#  Convert a netflow file to a CSV files with 4 tuples  


suppressPackageStartupMessages(library(optparse))
source("code/R/functins/convert_net24tuples.R")

option_list <- list(
  make_option("--input", action="store", type="character", help = "Set the name of the input netflow file"),
  make_option("--output", action="store", type="character", default="netflow-igraph.csv", help = "Set the name of the output  csv file")
)
opt <- parse_args(OptionParser(option_list=option_list))

if (opt$input %>% is.null() || opt$output %>% is.null()){
  message("[] Parameters missing. Please use --help for look at available parameters.")
  quit()
}else{
  df_4tuples <- convert_net24tuples(input_file = opt$input)
  ## Save 4 tuples files
  dir.create(dirname(opt$output), showWarnings = FALSE)
  readr::write_csv(df_4tuples,file=opt$output)
}