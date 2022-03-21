suppressPackageStartupMessages(library(optparse))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(igraph))
suppressPackageStartupMessages(library(visNetwork, quietly = TRUE, warn.conflicts = FALSE, verbose = FALSE))
suppressPackageStartupMessages(library(htmlwidgets, quietly = TRUE, warn.conflicts = FALSE, verbose = FALSE))

option_list <- list(
  make_option("--input", action="store", type="character", help = "Set the name of the input ncol file"),
  make_option("--output", action="store", type="character", default="igraph.png", help = "Set the name of the output  plot file")
)
opt <- parse_args(OptionParser(option_list=option_list))

if (is.null(opt$input) || is.null(opt$output)){
  message("[] Parameters missing. Please use --help for look at available parameters.")
  quit()
}else{
  message("[R] Reading `ncol` file ", opt$input ," as dataframe.")
  #net<- igraph::read_graph(opt$input,format='ncol')
  net_df <- readr::read_delim(opt$input, delim= " ", col_names = F, 
                              col_types = cols(col_character(),
                                               col_character(),
                                               col_double())
  )
  
  message("[R] Sampling 0.01")
  net_df <- net_df %>% sample_frac(0.01)
  message("[R] Total links in sample: ",nrow(net_df))
  net <- igraph::graph_from_data_frame(d=net_df, directed = T)
  V(net)$color = "blue"
  #png(opt$output,1200,1200)
  message("[R] Generating plot")
  #pdf(opt$output,width = 10, height = 10,compress = TRUE)
  svg(opt$output,width = 10, height = 10)
  layout = layout.drl
  #V(net)$size = degree(net) /max(degree(net))
  graph_plot<-plot(net, 
       rescale = TRUE,
       edge.arrow.size=0.001,
       edge.width= 0.001,
       #edge.width=E(net)$weight,
       vertex.label=NA,
       edge.curved=1, 
        layout = layout,
    # layout=layout.fruchterman.reingold,
      # asp =9/16
       )
  dev.off()
  message("[R] Saving pdf in: ",opt$output)
  message("[R] Generating htmlwidgets")
  data <- toVisNetworkData(net)
  data$nodes$font.size<-rep(0,data$nodes %>% nrow())
  saveWidget(
    visNetwork(nodes = data$nodes, edges = data$edges, height = "800px") %>%
      visIgraphLayout(layout = "layout.drl") %>%
      visEdges(width = 0.01)
    ,
    file = paste0(opt$output,".html")
    )
  message("[R] Saving htmlwidgets in : ",paste0(opt$output,".html"))
 
#  saveWidget(visIgraph(net)$font.size=c(0,0,0,0), )
  
  
}