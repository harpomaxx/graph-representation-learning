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
calculate_cluster_kmeans <- function (features_data,labels_data,nodes_data,k) {
  start <- Sys.time()
  message("[R] Calculating kmeans for k=", k)
  
  cluster_data <- kmeans(features_data, centers = k,nstart = 10, iter.max = 1000)
  features_cluster<-data.frame(node=nodes_data,features_data,label=labels_data,cluster=cluster_data$cluster)
  
  clusters_size <- features_cluster %>% group_by(cluster) %>% count()
  begnin_cluster<-clusters_size  %>% arrange(desc(n)) %>% head(1) %>% select(cluster) %>% unlist() %>% unname()
  tot_bot<-features_cluster %>% filter(label== "infected") %>% nrow()
  tot_host<-features_cluster %>% filter(label!= "infected") %>% nrow
  clust_bot<-features_cluster %>% filter(cluster == begnin_cluster) %>% filter(label=="infected") %>% nrow()
  clust_host<-features_cluster %>% filter(cluster == begnin_cluster) %>% filter(label!="infected") %>% nrow()
  bob <- tot_bot - clust_bot
  hob <- tot_host - clust_host
  
  stop <- Sys.time()
  message("[R] Total time elapsed: ", difftime(stop, start, units = "hour"))
  res<-list("bob"= bob,
            "hob"= hob,
            "k" = k,
            "kmeans"= cluster_data,
            "features_clustered" = features_cluster)
}
