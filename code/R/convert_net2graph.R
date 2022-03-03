suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))


#' Convert Netflow data file to pajek graph format
#'
#' @param input_file a neflow file
#' @return a dataframe with 4 tuples
#' @export
#'
#' @examples
#' 
convert_net2graph <- function (input_file){
  
  message("[] Reading dataset ",input_file)
  ctu_netflow <-
    read_csv(input_file)
  message("[] ",nrow(ctu_netflow)," rows read")
  ctu_netflow_filtered <- 
    ctu_netflow %>% 
    filter(Proto %in% c("udp", "tcp")) %>% 
    mutate(DstBytes = TotBytes-SrcBytes) %>%
    select(SrcAddr, DstAddr, DstBytes, SrcBytes) 
  
  message("[] Filtering tcp and udp: ",
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
  links <- ctu_netflow_all %>% select(origin,destination,weight) 
  net <- graph_from_data_frame(d=links, directed = T ) 
  ## Weights
  E(net)$weight <-links %>% select(weight)  %>% unname() %>% unlist()
  net
}

