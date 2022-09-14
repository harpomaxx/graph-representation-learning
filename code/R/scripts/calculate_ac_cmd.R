#!/bin/Rscript
#  calculate graph features  

suppressPackageStartupMessages(library(optparse))
suppressPackageStartupMessages(library(igraph))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(yaml))



source("code/R/functions/calculate_features.R")
source("code/R/functions/normalize_feature.R")

option_list <- list(
  make_option("--input", action="store", type="character", help = "Set the name of the input ncol file"),
  make_option("--output", action="store", type="character", default="netflow-features.csv", help = "Set the name of the output  csv file"),
  make_option("--outputnorm", action="store", type="character", default="netflow-features.csv", help = "Set the name of the output  csv file"),
  make_option("--depth", action="store", type="integer", default="1", help = "Set the order of the neibourhood ")
)
opt <- parse_args(OptionParser(option_list=option_list))

if (opt$input %>% is.null() || opt$output %>% is.null() ||
    opt$depth %>% is.null() || opt$outputnorm %>% is.null()
    ){
  message("[] Parameters missing. Please use --help for look at available parameters.")
  quit()
}else{
  net_graph<-read_graph(opt$input, format='ncol', directed = TRUE)
  ## Set default parameters
  params <- yaml::read_yaml("params.yaml")
  if(!  "calculate_ac" %in% names(params)) {
    message("[] Error: no information found")
    quit()
  }
  ##  Required for Normalize
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
  
 
  # create directories 
  dir.create(dirname(opt$output), showWarnings = FALSE, recursive = TRUE)
  dir.create(dirname(opt$outputnorm), showWarnings = FALSE, recursive = TRUE)

  for (alpha in params$calculate_ac$alphas){
      ac_f <- calculate_ac(net_graph,alpha=alpha)
      ac_f <- ac_f %>% as.data.frame()  %>% 
        tibble::add_column(node=net_graph %>% 
                         get.vertex.attribute('name')) %>% rename(AC=ac) %>% select(node,AC)
      start <- Sys.time()
      message("[R] normalize feature ac")
      ac_f_norm  <- 
          sapply(seq(1, length(V(net_graph))), function(x)
        normalize_feature(featureTarget = ac_f %>% tibble::remove_rownames(), neighbourhood, k, x))
  
      stop <- Sys.time()
      message("[R] Total time elapsed: ", difftime(stop, start, units = "hour"))
  
      ac_f_norm <- ac_f_norm %>% as.data.frame()  %>% 
        tibble::add_column(node=net_graph %>% 
                         get.vertex.attribute('name')) %>% rename(AC=".") %>% select(node,AC)
  
  
  ## Save features
      write_csv(ac_f,file = paste0(opt$output,"-alpha=",alpha,".ac"))
      write_csv(ac_f_norm,file = paste0(opt$outputnorm,"-alpha=",alpha,".ac"))
  }
  write(x = "",paste0("ac.",Sys.getpid(),".end"))
}
