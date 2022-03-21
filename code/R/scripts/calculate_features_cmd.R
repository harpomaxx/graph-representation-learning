#!/bin/Rscript
#  calculate graph features  

suppressPackageStartupMessages(library(optparse))
suppressPackageStartupMessages(library(igraph))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))



source("code/R/functions/calculate_bc.R")

option_list <- list(
  make_option("--input", action="store", type="character", help = "Set the name of the input ncol file"),
  make_option("--output", action="store", type="character", default="netflow-features.csv", help = "Set the name of the output  csv file")
)
opt <- parse_args(OptionParser(option_list=option_list))

if (opt$input %>% is.null() || opt$output %>% is.null()){
  message("[] Parameters missing. Please use --help for look at available parameters.")
  quit()
}else{
  #net_df <- readr::read_delim(opt$input, delim= " ", col_names = F, 
  #                            col_types = cols(col_character(),
  #                                             col_character(),
  #                                             col_double())
  #)
  #message("[R] ",nrow(net_df), " rows read")
  #net_df <- net_df %>% filter(X3 != 0)
  #names(net_df)<-c("v1","v2","weight")
  #message("[R] ",nrow(net_df), " rows remains")
  
  #net_graph <- igraph::graph_from_data_frame(d=net_df,  directed = T)
  #message("[R] Is graph weighted:", is_weighted(net_graph))
  
  net_graph<-read_graph(opt$input, format='ncol')
  bc_f <- calculate_bc(net_graph)
  ## Save features
  dir.create(dirname(opt$output), showWarnings = FALSE)
  write_csv(bc_f %>% as.data.frame()  %>% tibble::rownames_to_column("node")
            ,file = opt$output)
}