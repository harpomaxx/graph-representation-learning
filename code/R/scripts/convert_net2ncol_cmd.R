#!/bin/Rscript
#  Convert a netflow file to a `ncol`file

suppressPackageStartupMessages(library(optparse))
suppressPackageStartupMessages(library(readr))

source("code/R/functions/convert_net2ncol.R")

option_list <- list(
  make_option("--input", action="store", type="character", help = "Set the name of the input netflow file"),
  make_option("--output", action="store", type="character", default="netflow-igraph.ncol", help = "Set the name of the output  ncol file")
)
opt <- parse_args(OptionParser(option_list=option_list))

if (opt$input %>% is.null() || opt$output %>% is.null()){
  message("[] Parameters missing. Please use --help for look at available parameters.")
  quit()
}else{
  df_ncol <- convert_net2ncol(input_file = opt$input)
  message("[R] ",nrow(df_ncol)," edges found")
  
  dir.create(dirname(opt$output), showWarnings = FALSE)
  write_delim(df_ncol, file=opt$output, delim = " ", col_names = FALSE)
  message("[R] ncol file saved in", opt$output)
  
  message("[R] Removing edges with zero weights")
  df_ncol<-df_ncol %>% filter( weight != 0)
  message("[R] ",nrow(df_ncol)," edges found")
  
  output_pos_weights<-paste0(sub('\\..*$', '', opt$output),"-positive-weights.ncol")
  write_delim(df_ncol, file=output_pos_weights , delim = " ", col_names = FALSE)
  message("[R] ncol file without zeros saved in", output_pos_weights)
  

  #write_graph(net_graph,file=opt$output,format="ncol")
}
