suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(igraph))
suppressPackageStartupMessages(library(tibble))

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
  igraph_obj <- igraph::as.undirected(igraph_obj, mode = "collapse")
  
  lcc <-
    igraph::transitivity(igraph_obj,
                         type = "local",
                         isolates = "zero" ,
                         weights = NULL)
  stop <- Sys.time()
  message("[R] Total time elapsed: ", difftime(stop, start, units = "hour"))
  lcc
}

#' calculate degree for igraph object
#'
#' @param igraph_obj a igraph_obj file
#' @return
#' @export
#'
#' @examples
#'
calculate_degree <- function (igraph_obj) {
  start <- Sys.time()
  message("[R] Calculating Degree")
  igraph_obj <- igraph::as.directed(igraph_obj)
  degree <-
    list(
      "degree_in" = unname(degree(igraph_obj, V(igraph_obj) , mode = "in")),
      "degree_out" = unname(degree(igraph_obj, V(igraph_obj), mode = "out")),
      "strength_in" = unname(strength(igraph_obj, V(igraph_obj), mode = "in")),
      "strength_out" = unname(strength(igraph_obj, V(igraph_obj), mode = "out"))
    )
  stop <- Sys.time()
  message("[R] Total time elapsed: ", difftime(stop, start, units = "hour"))
  degree
}

calculate_features <- function(igraph_obj) {
  features_f<-cbind(as.data.frame(calculate_lcc(igraph_obj)),
        as.data.frame(calculate_degree(igraph_obj)))
  names(features_f) <-c("LCC","ID","OD","IDW","ODW")
  features_f
}