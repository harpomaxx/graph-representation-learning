suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(igraph))
suppressPackageStartupMessages(library(doMC))


#' calculate BC data file to graph
#'
#' @param igraph_obj a igraph_obj file
#' @return
#' @export
#'
#' @examples
#'
calculate_bc <- function (igraph_obj) {
  #print(igraph_obj, v=TRUE)
  registerDoMC(10)
  start <- Sys.time()
  message("[R] WARNING! Removing edges with zero weight")
  igraph_obj<-delete_edges(igraph_obj, which(E(igraph_obj)$weight==0))
  message("[R] Calculating BC")
  bc <- betweenness(
    igraph_obj,
    v = V(igraph_obj),
    directed = TRUE,
    weights = E(igraph_obj)$weight,
    #cutoff = 0,
    normalized = TRUE
  )
  stop <- Sys.time()
  message("[R] Total time elapsed: ", stop - start)
  bc
}

