#!/bin/Rscript

suppressPackageStartupMessages(library(optparse))
suppressPackageStartupMessages(library(igraph))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
source("code/R/functions/normalize_feature.R")

option_list <- list(
  make_option("--featuresfile", action="store", type="character", help = "Set the name of the input csv with features values"),
  make_option("--ncolfile", action="store", type="character", default="ncolfile", help = "name of the input file with graph information in ncol format"),
  make_option("--output", action="store", type="character", default="netflow-features-normalized.csv", help = "Set the name of the output  csv file with features normalized"),
  make_option("--depth", action="store", type="integer", default="1", help = "Set the order of the neibourhood ")
)
opt <- parse_args(OptionParser(option_list=option_list))

if (opt$featuresfile%>% is.null() || opt$output %>% is.null() || opt$ncolfile %>% is.null()){
  message("[] Parameters missing. Please use --help for looking at available parameters.")
  quit()
}else{
  message("[R] Normalizing ", opt$featuresfile," for order ", opt$depth)
  net_graph <- read_graph(opt$ncolfile, format = 'ncol', directed = TRUE)
  
  features <-
    readr::read_csv(opt$featuresfile, col_types = cols())
  
  neighbourhood <-
    ego(
      net_graph,
      order = opt$depth,
      nodes = V(net_graph),
      mode = "all",
      mindist = 1
    )
  names(neighbourhood) <- V(net_graph)$name
  k <-
    ego_size(
      net_graph,
      order = opt$depth,
      nodes = V(net_graph),
      mode = "all",
      mindist = 1
    )
  
  features_names <- colnames(features)[2:ncol(features)]
  features_n <- list()
  start <- Sys.time()
  for (feature in features_names) {
    message("[R] normalize feature ",feature)
    featuretarget <- features |> select(node, all_of(feature))
    features_n[[feature]] <-
      sapply(seq(1, length(V(net_graph))), function(x)
        normalize_feature(featureTarget = featuretarget, neighbourhood, k, x))
  }
  stop <- Sys.time()
  message("[R] Total time elapsed: ", difftime(stop, start, units = "hour"))
  
  features_n <- features_n %>% as.data.frame() %>% tibble::add_column(node=features$node) %>% select(names(features))
  
  
  ## Save features
  dir.create(dirname(opt$output), showWarnings = FALSE, recursive = TRUE)
  write_csv(features_n
            ,file = opt$output)
  write(x = "",paste0("fnormalize",Sys.getpid(),".end"))
  message("[R] f-normalized file saved in ", opt$output)
}