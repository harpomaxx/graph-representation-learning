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


#' calculate alpha-centrality for igraph object
#'
#' @param igraph_obj a igraph_obj file
#' @return
#' @export
#'
#' @examples
#'
calculate_ac <- function (igraph_obj,alpha = 0.01) {
  start <- Sys.time()
  message("[R] Calculating ac for alpha = ",alpha )
  ac <-
    list(
      "ac" = unname(igraph::alpha.centrality(igraph_obj,
                                                    alpha = alpha,
                                                    exo = 1,
                                                    weights = NULL
                                                    ))
    )
  stop <- Sys.time()
  message("[R] Total time elapsed: ", difftime(stop, start, units = "hour"))
  ac
}


calculate_features <- function(igraph_obj) {
  features_f<-cbind(
        as.data.frame(calculate_lcc(igraph_obj)),
        as.data.frame(calculate_degree(igraph_obj)),
        #as.data.frame(calculate_ac(igraph_obj))
        )
  names(features_f) <-c("LCC","ID","OD","IDW","ODW")
  features_f
}
