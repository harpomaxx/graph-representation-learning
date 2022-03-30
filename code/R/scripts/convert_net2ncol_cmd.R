#!/bin/Rscript
#  Convert a netflow file to a `ncol`file

suppressPackageStartupMessages(library(optparse))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(yaml))

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
  df_ncol_n <- nrow(df_ncol)
  df_ncol_vertex <- rbind(df_ncol$origin,df_ncol$destination) %>% unique() %>% length()
  message("[R] ",df_ncol_n," edges found")
  message("[R] ",df_ncol_vertex," vertex found")
  
  
  dir.create(dirname(opt$output), showWarnings = FALSE)
  write_delim(df_ncol, file=opt$output, delim = " ", col_names = FALSE)
  message("[R] ncol file saved in", opt$output)
  
  message("[R] Removing edges with zero weights")
  df_ncol<-df_ncol %>% filter( weight != 0)
  df_ncol_final_n <- nrow(df_ncol)
  df_ncol_final_vertex <- rbind(df_ncol$origin,df_ncol$destination) %>% unique() %>% length()
  
  message("[R] ",df_ncol_final_n," edges found")
  df_ncol_wmean <- mean(df_ncol$weight)
  df_ncol_wsd <- sd(df_ncol$weight)
  
  output_pos_weights<-paste0(sub('\\.ncol', '', opt$output),"-positive-weights.ncol")
  write_delim(df_ncol, file=output_pos_weights , delim = " ", col_names = FALSE)
  message("[R] ncol file without zeros saved in", output_pos_weights)
  
  l <- list()
  name <-sub('\\.binetflow.labels.ncol','',basename(opt$output))
  
  #l[[name]] <- list(
  l <- list("edges" = df_ncol_n,
            "nzero_edges" = df_ncol_final_n,
            "vertex" = df_ncol_vertex,
            "nzero_vertex" = df_ncol_vertex,
            "nzero_wmean" =  df_ncol_wmean,
            "nzero_wsd" =  df_ncol_wsd)
  
  l %>% as.yaml() %>% 
         write(paste0("metrics/",name,"_ncol.yaml"))
  
  
  

  #write_graph(net_graph,file=opt$output,format="ncol")
}
