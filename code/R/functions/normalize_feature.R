suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(igraph))
suppressPackageStartupMessages(library(tibble))


#' Normalize feature 
#' Given a data.frame of a certain feature, the set of neighbors and the size of each neighborhood for any vertex, 
#' the normalization is calculated for the vertex located at the position indicated by the index in the vertex list.
#'  Note 1: this function is to be used with the *apply family 
#' Note 2: "Feature Normalization" (see more in Daya et.al 2020) consists of:
#' The neighborhood set $N_i$ for vertex $v_i \in V$ is restricted to depth $D=1$.
#' The mean of feature for vertex $v_i$ across its neighbors $v_k \in N_i$ are computed.
#' Thus, feature relative to their neighborhood mean is given as:
#'       $\mu_{i,m} = \frac{\sum_{v_k \in N_i} f_{k,m}}{\abs{N_i}}$
#'       $f_{i,m} = \frac{f_{i,m}}{\mu_{i,m}}$
#'       For all $v_i \in V$ and $0 \leq m \leq 6$
#'       
#' @param featureTarget data.frame with columns "node" and the feature of interest (e.g. "ID", "OD", "IDW", "ODW", "BC", "LCC", "AC")
#' @param neighbourhood list of neighbours to certain depth (given by args[2]) for any vertex
#' @param k list of sizes of the corresponding neighbourhoods
#' @param index relative to the vertex at the position indicated by the index
#'
#' @return normalized value
#' @export
#'
#' @examples
normalize_feature <- function(featureTarget, neighbourhood, k, index) {
  start <- Sys.time()
  #smessage("[R] normalize feature ",names(featureTarget)[2])
  numberNeighbours <- k[index]
  localNeighbourhood <- neighbourhood[index][[1]]
  suma <- 0
  for(i in localNeighbourhood) {
    suma <- suma + featureTarget[i, 2]
  }
  stop <- Sys.time()
  #message("[R] Total time elapsed: ", difftime(stop, start, units = "hour"))
  
  if(suma != 0) {
    mu <- suma / numberNeighbours
    return(as.numeric(featureTarget[index,2]/mu))
  } else {
    return(as.numeric(featureTarget[index,2]))
  }
}
 
  
