suppressPackageStartupMessages(library(optparse))
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
  net<- igraph::read_graph(opt$input,format='ncol')
  V(net)$color = "blue"
  #png(opt$output,1200,1200)
  pdf(opt$output,width = 10, height = 10,compress = TRUE)
  layout = layout.drl
  V(net)$size = degree(net) /max(degree(net))
  graph_plot<-plot(net, 
       edge.arrow.size=0.01,
       edge.width= 0.01,
       #edge.width=E(net)$weight,
       vertex.label=NA,
       edge.curved=1, 
        layout = layout,
    # layout=layout.fruchterman.reingold,
      # asp =9/16
       )
  dev.off()
  data <- toVisNetworkData(net)
  data$nodes$font.size<-rep(0,data$nodes %>% nrow())
  saveWidget(
    visNetwork(nodes = data$nodes, edges = data$edges, height = "800px") %>%
      visIgraphLayout(layout = "layout.drl") %>%
      visEdges(width = 0.01)
    
    ,
    file = paste0(opt$output,".html")
    )
 
#  saveWidget(visIgraph(net)$font.size=c(0,0,0,0), )
  
  
}