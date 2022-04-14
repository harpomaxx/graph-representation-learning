#!/bin/Rscript

suppressPackageStartupMessages(library(optparse))
suppressPackageStartupMessages(library(igraph))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))



option_list <- list(
  make_option("--allfeaturesfile", action="store", type="character", help = "Set the name of the input csv with features vaules"),
  make_option("--bcfeaturefile", action="store", type="character", default="bcfile.csv", help = "name of the input file with node and the BC feature"),
  make_option("--output", action="store", type="character", default="netflow-features.csv", help = "Set the name of the output  csv file")
)
opt <- parse_args(OptionParser(option_list=option_list))

if (opt$allfeaturesfile%>% is.null() || opt$output %>% is.null() || opt$bcfeaturefile %>% is.null()){
  message("[] Parameters missing. Please use --help for look at available parameters.")
  quit()
}else{
  message("[R] Merging ", opt$allfeaturesfile, " and ", opt$bcfeaturefile)
  features<- readr::read_csv(opt$allfeaturesfile, col_types = cols())
  bc_feature <- readr::read_csv(opt$bcfeaturefile, col_types = cols())
  names(bc_feature)<-c("node","BC")
  features_f<-cbind(bc_feature,features)
  ## Save features
  dir.create(dirname(opt$output), showWarnings = FALSE, recursive = TRUE)
  write_csv(features_f
            ,file = opt$output)
  write(x = "",paste0("merged.",Sys.getpid(),".end"))
  message("[R] Merged file saved in ", opt$output)
}