suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(igraph))

#' calculate features data file to graph
#'
#' @param igraph_obj a igraph_obj file
#' @return
#' @export
#'
#' @examples
#'
calculate_lcc <- function (igraph_obj) {
  start <- Sys.time()
  message("[R] Calculating LCC")
  igraph_obj <- igraph::as.undirected(igraph_obj, mode="collapse")
  
  lcc <- igraph::transitivity(igraph_obj, type = "local", isolates = "zero")
  stop <- Sys.time()
  message("[R] Total time elapsed: ",difftime(stop,start,units = "hour"))
  lcc
}

calculate_features <- function(igraph_obj){
  calculate_lcc(igraph_obj)
}