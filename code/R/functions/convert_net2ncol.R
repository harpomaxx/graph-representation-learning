suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
#suppressPackageStartupMessages(library(igraph))

#' Convert Netflow data file to ncol format using bytes as graph weights
#'
#' @param input_file a neflow file
#' @return dataframe in ncol format
#' @export
#'
#' @examples
#' 
convert_net2ncol <- function (input_file){
  options(dplyr.summarise.inform = FALSE)
  message("[R] Reading dataset ",input_file)
  ctu_netflow <-
    read_csv(input_file)
  message("[R] ",nrow(ctu_netflow)," rows read")
  ctu_netflow_filtered <- 
    ctu_netflow %>% 
    filter(Proto %in% c("udp", "tcp")) %>% 
    mutate(DstBytes = TotBytes-SrcBytes) %>%
    select(SrcAddr, DstAddr, DstBytes, SrcBytes) 
  
  message("[R] Filtering tcp and udp: ",
          nrow(ctu_netflow_filtered), " rows remains")
  # create src_dst and dst_src keys
  ctu_netflow_srcdst <- ctu_netflow_filtered %>% 
    select(SrcAddr, DstAddr, SrcBytes) %>% 
    rename(origin=SrcAddr,destination="DstAddr",weight="SrcBytes")
  
  ctu_netflow_dstsrc<-ctu_netflow_filtered %>% 
    select(DstAddr, SrcAddr, DstBytes) %>% 
    rename(origin=DstAddr,destination="SrcAddr",weight="DstBytes")
  
  ctu_netflow_all<-rbind(ctu_netflow_dstsrc,ctu_netflow_srcdst) 
  ctu_netflow_all<- ctu_netflow_all %>% group_by(origin,destination) %>% 
    summarise(weight=sum(weight)) %>% ungroup()
  ctu_netflow_all %>% select(origin,destination,weight) 
  #net <- graph_from_data_frame(d=links, directed = T ) 
  ## Weights
  #E(net)$weight <-links %>% select(weight)  %>% unname() %>% unlist()
  #net
}

#' Convert Netflow data file to ncol format using pkts as graph weights
#'
#' @param input_file a neflow file
#' @return dataframe in ncol format
#' @export
#'
#' @examples
#' 
convert_net2ncol_pkts <- function (input_file){
  options(dplyr.summarise.inform = FALSE)
  message("[R] Reading dataset ",input_file)
  ctu_netflow <-
    read_csv(input_file)
  message("[R] ",nrow(ctu_netflow)," rows read")
  ctu_netflow_filtered <- 
    ctu_netflow %>% 
    filter(Proto %in% c("udp", "tcp")) %>% 
    select(SrcAddr, DstAddr, DstPkts, SrcPkts) 
  
  message("[R] Filtering tcp and udp: ",
          nrow(ctu_netflow_filtered), " rows remains")
  # create src_dst and dst_src keys
  ctu_netflow_srcdst <- ctu_netflow_filtered %>% 
    select(SrcAddr, DstAddr, SrcPkts) %>% 
    rename(origin=SrcAddr,destination="DstAddr",weight="SrcPkts")
  
  ctu_netflow_dstsrc<-ctu_netflow_filtered %>% 
    select(DstAddr, SrcAddr, DstPkts) %>% 
    rename(origin=DstAddr,destination="SrcAddr",weight="DstPkts")
  
  ctu_netflow_all<-rbind(ctu_netflow_dstsrc,ctu_netflow_srcdst) 
  ctu_netflow_all<- ctu_netflow_all %>% group_by(origin,destination) %>% 
    summarise(weight=sum(weight)) %>% ungroup()
  ctu_netflow_all %>% select(origin,destination,weight) 
}


