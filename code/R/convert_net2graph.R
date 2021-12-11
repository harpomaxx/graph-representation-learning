#!/bin/Rscript
#  Convert a netflow file to pajek file format using igraph library

suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(igraph))
suppressPackageStartupMessages(library(optparse))

#' Convert Netflow data file to pajek graph format
#'
#' @param input_file 
#' @param output_file 
#'
#' @return
#' @export
#'
#' @examples
#' 
convert_net2graph <- function (input_file, output_file){
  
  ctu_netflow <-
    read_csv(input_file)
  ctu_netflow_filtered <-
    ctu_netflow %>% filter(Proto %in% c("udp", "tcp")) %>% 
    select(SrcAddr, DstAddr, TotPkts) %>% 
    tidyr::unite("src_dst", c("SrcAddr", "DstAddr"))
  ctu_netflow_src_dst_agg <-
    ctu_netflow_filtered %>% group_by(src_dst) %>% 
    summarise(TotPkts = sum(TotPkts))
  ctu_netflow_agg <-
    ctu_netflow_src_dst_agg %>% 
    tidyr::separate(src_dst, c("src", "dst"), sep = "_")
  #nodes <- c(ctu_netflow_agg$src , ctu_netflow_agg$dst) %>% unique()
  links <-
    ctu_netflow_agg %>% select(src, dst, TotPkts) %>% unique()
  net <-
    graph_from_data_frame(d = links,
                          directed = T)
  E(net)$weight <-
    links %>% select(TotPkts)  %>% unname() %>% unlist()
  write_graph(net, file = output_file, format = "pajek")
}

#### MAIN 

option_list <- list(
  make_option("--input", action="store", type="character", help = "Set the name of the input netflow file"),
  make_option("--output", action="store", type="character", default="netflow-igraph.dot", help = "Set the name of the output  pajek file")
)
opt <- parse_args(OptionParser(option_list=option_list))

if (opt$input %>% is.null() || opt$output %>% is.null()){
  message("[] Parameters missing. Please use --help for look at available parameters.")
  quit()
}else{
  convert_net2graph(input_file = opt$input, output_file = opt$output)
}

