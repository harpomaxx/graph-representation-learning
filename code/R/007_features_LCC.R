#!/bin/Rscript


# Note: code adapted from https://github.com/jgrapht/jgrapht/blob/6aba8e81053660997fe681c50974c07e312027d1/jgrapht-core/src/main/java/org/jgrapht/alg/scoring/ClusteringCoefficient.java#L168


suppressPackageStartupMessages(library(igraph))
suppressPackageStartupMessages(library(dplyr))

args <- commandArgs(TRUE)
ncolName <- paste(args[1], "_noZeroB.ncol", sep = "")
csvName <- paste(args[1], "_LCC.csv", sep = "")

sprintf("CAPTURE: %s", args[1])
sprintf("Create graph from: %s", ncolName)
sprintf("Store local clustering coefficient at: %s", csvName)


computeLocalClusteringCoefficient <- function(graph, vertex) {
    # Calculate LCC for a vertex in graph
   
    neighbourhood <- ego(graph, order = 1, nodes = vertex, mode = "all", mindist = 1) 
    k <- ego_size(graph, order = 1, nodes = vertex, mode = "all", mindist = 1) 
    # https://igraph.org/r/doc/ego.html
    # see (1)
    
    numberTriplets <- 0
    
    for (p in as.vector(neighbourhood[[1]])) {
        for (q in as.vector(neighbourhood[[1]])) {
            if (are_adjacent(graph, p, q)) {
                # https://igraph.org/r/doc/are_adjacent.html
                # see (2)
                numberTriplets <- numberTriplets + 1
            }
        }
    }
    
    if (k <= 1) {
        return(0.0)
    } else {
        return(numberTriplets / (k * (k - 1)))
    }
}


##### Load graph #####
start_load <- proc.time()
g <- read_graph(file = ncolName, format = "ncol", directed = T)
end_load <- proc.time()

time_load = end_load - start_load
print(" time_load : ")
print(time_load)


##### Calculate Local Clustering Coefficient (LCC) #####
start_LCC <- proc.time()
LCC <- lapply(V(g), function(x) computeLocalClusteringCoefficient(g,x))
end_LCC <- proc.time()

time_LCC = end_LCC - start_LCC
print(" time_LCC : ")
print(time_LCC)


##### Create a DataFrame and store it in csv file #####
start_store <- proc.time()
df <- data.frame(LCC) %>% as_tibble(rownames = "node")
write.csv(df, csvName, row.names = FALSE)
end_store <- proc.time()

time_store = end_store - start_store
print(" time_store : ")
print(time_store)



##### About the original code #####

# (1)
# The neighbourhood of a vertex is calculated from "class NeighborCache":
# https://github.com/jgrapht/jgrapht/blob/master/jgrapht-core/src/main/java/org/jgrapht/alg/util/NeighborCache.java#L94

# "neighborListOf" method of "class Graphs" is used for this purpose:
# https://github.com/jgrapht/jgrapht/blob/6aba8e81053660997fe681c50974c07e312027d1/jgrapht-core/src/main/java/org/jgrapht/Graphs.java#L271

# And for this, the "edgesOf" method is used. In case of directed graphs:
# https://github.com/jgrapht/jgrapht/blob/master/jgrapht-core/src/main/java/org/jgrapht/graph/specifics/DirectedSpecifics.java#L196


# (2)
# https://github.com/jgrapht/jgrapht/blob/6aba8e81053660997fe681c50974c07e312027d1/jgrapht-core/src/main/java/org/jgrapht/Graph.java#L273

