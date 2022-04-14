#!/bin/Rscript
#  label nodes

suppressPackageStartupMessages(library(optparse))
suppressPackageStartupMessages(library(igraph))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))



source("code/R/functions/label_dataset.R")

option_list <- list(
  make_option("--input", action="store", type="character", help = "Set the name of the input csv file"),
  make_option("--labelsfile", action="store", type="character", default="label.csv", help = "Set the name of the file used for labeling"),
  make_option("--output", action="store", type="character", default="netflow-features.csv", help = "Set the name of the output  csv file")
)
opt <- parse_args(OptionParser(option_list=option_list))

if (opt$input %>% is.null() || opt$output %>% is.null() || opt$labelsfile %>% is.null()){
  message("[] Parameters missing. Please use --help for look at available parameters.")
  quit()
}else{
  features <- readr::read_csv(opt$input,col_names = TRUE, col_types = cols())
  labels <- readr::read_csv(opt$labelsfile,col_names = FALSE, col_types = cols())
  message("[R] Labeling dataset: ", opt$input, " with ", opt$labelsfile)
  features_w_labels <- label_dataset(features,labels)
  ## Save features
  dir.create(dirname(opt$output), showWarnings = FALSE, recursive = TRUE)
  write_csv(features_w_labels
            ,file = opt$output)
  write(x = "",paste0("labeled.",Sys.getpid(),".end"))
  message("[R] File stored in ", opt$output)
}