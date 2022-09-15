#!/bin/Rscript
#  Script for tuning the bast alpha value when used with kmeans clustering 
#  algorithms 

## Only valid for CTU13 dataset. A lot of information about captures are 
## hard-coded

suppressPackageStartupMessages(library(optparse))
suppressPackageStartupMessages(library(igraph))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(yaml))
suppressPackageStartupMessages(library(doMC))
registerDoMC(cores=7)

source("code/R/functions/calculate_clustering.R")

option_list <- list(
  make_option("--labeldir", action="store", type="character", help = "Set the name directory where labeled datasets are stored"),
  make_option("--acdir", action="store", type="character", help = "Set the name directory where ac results are stored")
)
opt <- parse_args(OptionParser(option_list=option_list))

if (opt$labeldir %>% is.null() || opt$acdir %>% is.null()){
  message("[] Parameters missing. Please use --help for look at available parameters.")
  quit()
}else{

  
  ## Set default parameters
  params <- yaml::read_yaml("params.yaml")
  if(!  ("tune_best_ac_for_kmeans" %in% names(params) 
     && "calculate_ac" %in% names(params))) {
    message("[] Error: no parameters information found")
    quit()
  }
  
  capture_list<-list.files(opt$labeldir,pattern = "capture2011081[012345689](-[123])?.binetflow.labels",full.names = T)
  #capture_list <- params$tune_best_ac_for_kmeans$capture_list
  message("[R] Reading ncold files") 
  print(capture_list)
  
  for (alpha in params$calculate_ac$alphas){
    message("[R] Calculating centroids for alpha=",alpha )
    features_data<-list()
    for (capture in capture_list){
      capture_df <- readr::read_csv(capture,col_types = cols())
      capture_df <- capture_df %>% select(-AC)
      # reading new AC
      capture_base <- stringr::str_split(capture,pattern='\\.')[[1]][1]
      acfile <- paste0(opt$acdir,basename(capture_base),".binetflow.labels-positive-weights-alpha=",alpha,".ac")
      print(acfile)
      ac_df <- readr::read_csv(
        paste0(opt$acdir,basename(capture_base),".binetflow.labels-positive-weights-alpha=",alpha,".ac"),col_types = cols()
      )
      capture_df<-capture_df %>% tibble::add_column(AC=ac_df %>% select(AC) %>% unlist()) %>% select(node,ID,OD,IDW,ODW,BC,LCC,AC,label)
      features_data<-rbind(features_data,capture_df)
    }
    labels_data <- features_data %>% select(label)
    nodes_data <- features_data %>% select(node)
    features_data<-features_data %>% select(-label,-node)
    res <- list()
    
    centroids <- params$tune_best_ac_for_kmeans$centroids
    res <- foreach(i = 1:length(params$tune_best_ac_for_kmeans$centroids), .combine = rbind) %dopar% {
      k = centroids[i]
      cluster_res <- calculate_cluster_kmeans(features_data,labels_data,nodes_data, k = k)
      partial_res <- c(k,cluster_res$bob,cluster_res$hob)
    }
    
    #for (k in params$tune_best_ac_for_kmeans$centroids){
    #  cluster_res <- calculate_cluster_kmeans(features_data,labels_data,nodes_data, k = k)
    #  res<-rbind(res,
    #             c(k,cluster_res$bob,cluster_res$hob))
    #}
    res<-res %>% as.data.frame()     
    names(res)<-c("k","bob","hob")
    metrics_dir<-paste0("metrics/kmeans/",basename(opt$acdir))
    dir.create(metrics_dir, showWarnings = FALSE, recursive = TRUE)
    res %>% yaml::as.yaml() %>% yaml::write_yaml(paste0(metrics_dir,"/kmeans_alpha_",alpha,".yaml"))
 
 }
  ## write(x = "",paste0("kmeans.",Sys.getpid(),".end"))
}

