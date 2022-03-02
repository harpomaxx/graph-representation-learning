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
convert_net24tuples <- function (input_file){
  
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
  
  # creating src_dst and dst_src keys
  ctu_netflow_srcdst <- ctu_netflow_filtered %>% 
    select(SrcAddr, DstAddr, SrcBytes, DstBytes) %>% 
    tidyr::unite("src_dst", c("SrcAddr","DstAddr")) 
  ctu_netflow_dstsrc <- ctu_netflow_filtered %>% 
    select(SrcAddr,DstAddr, SrcBytes, DstBytes) %>% 
    tidyr::unite("dst_src", c("DstAddr","SrcAddr")) 
  
  # aggregating Bytes from src_dst and dst_src
  ctu_netflow_src_dst_agg <- ctu_netflow_srcdst %>% 
    group_by(src_dst) %>% 
    summarise(TotSrcBytes = sum(SrcBytes),
              TotDstBytes = sum(DstBytes))
  ctu_netflow_dst_src_agg <- ctu_netflow_dstsrc %>% 
    group_by(dst_src) %>% 
    summarise(TotSrcBytes = sum(SrcBytes),
              TotDstBytes = sum(DstBytes))
  
  ctu_netflow_4tuple<-left_join(ctu_netflow_src_dst_agg,
                                ctu_netflow_dst_src_agg,
                                by=c("src_dst"="dst_src")) %>%
    mutate_all(~replace(., is.na(.), 0))
  
  message("[] Aggregating `src` and `dst` stats: ",
          nrow(ctu_netflow_4tuple), " rows remains")
  
  ctu_netflow_4tuple <-
    ctu_netflow_4tuple %>% 
    mutate(TotSrcBytes = TotSrcBytes.x + TotDstBytes.y,
           TotDstBytes = TotDstBytes.x + TotSrcBytes.y) %>%
    select(src_dst, TotSrcBytes, TotDstBytes) 
  
  message("[] Removing reverse duplicates")
  unique_ctu_netflow_4tuple <- 
    ctu_netflow_4tuple %>%  
    tidyr::separate(src_dst, c("SrcAddr", "DstAddr"), sep = "_") 
    group_by(grp = paste(pmax(SrcAddr, DstAddr), 
                         pmin(SrcAddr, DstAddr), 
                         sep = "_")) %>%
    slice(1) %>%
    ungroup() %>%
    select(-grp) %>% tidyr::unite("src_dst", c("SrcAddr", "DstAddr")) %>%
    tidyr::separate(src_dst, c("SrcAddr", "DstAddr"), sep = "_") %>%
    select(SrcAddr, DstAddr, TotSrcBytes, TotDstBytes) 
  message("[] ", 
          nrow(ctu_netflow_4tuple)-nrow(unique_ctu_netflow_4tuple), 
          " reverse duplicated removed")
  message("[] ", nrow(unique_ctu_netflow_4tuple)," links extracted")
  message("[] Done")
   unique_ctu_netflow_4tuple
}

