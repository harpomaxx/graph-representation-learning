#!/bin/Rscript


suppressPackageStartupMessages(library(igraph))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(readr))


args <- commandArgs(TRUE)
ncolName <- paste(args[1], "_noZeroB.ncol", sep = "")
featuresFile <- paste("features/", args[1], "_features.csv", sep = "")
csvName <- paste(args[1], "_features_normalized.csv", sep = "")
depth <- args[2]

cat("CAPTURE: ", args[1], "\n")
cat("Create graph from: ", ncolName, "\n")
cat("Store normalized features (depth = ", depth, ") at: ", csvName, "\n")


normalizeFeature <- function(featureTarget, neighbourhood, k, index) {
    # Given a data.frame of a certain feature, the set of neighbors and the size of each neighborhood for any vertex, 
    # the normalization is calculated for the vertex located at the position indicated by the index in the vertex list.
    #
    # Arguments: featureTarget: data.frame with columns "node" and the feature of interest (e.g. "ID", "OD", "IDW", "ODW", "BC", "LCC", "AC")
    #            neighbourhood: list of neighbours to certain depth (given by args[2]) for any vertex
    #            k: list of sizes of the corresponding neighbourhoods
    #            index: relative to the vertex at the position indicated by the index
    # Return: normalized value

    # Note 1: this function is to be used with the *apply family 
    #
    # Note 2: "Feature Normalization" (see more in Daya et.al 2020) consists of:
    # The neighborhood set $N_i$ for vertex $v_i \in V$ is restricted to depth $D=1$.
    # The mean of feature for vertex $v_i$ across its neighbors $v_k \in N_i$ are computed.
    # Thus, feature relative to their neighborhood mean is given as:
    #       $\mu_{i,m} = \frac{\sum_{v_k \in N_i} f_{k,m}}{\abs{N_i}}$
    #       $f_{i,m} = \frac{f_{i,m}}{\mu_{i,m}}$
    #       For all $v_i \in V$ and $0 \leq m \leq 6$
    #

    numberNeighbours <- k[index]
    localNeighbourhood <- neighbourhood[index][[1]]
    
    suma <- 0
    for(i in localNeighbourhood) {
        suma <- suma + featureTarget[i, 2]
    }
    
    if(suma != 0) {
        mu <- suma / numberNeighbours
        return(as.numeric(featureTarget[index,2]/mu))
    } else {
        return(as.numeric(featureTarget[index,2]))
    }
}


##### Load graph #####
start_load <- proc.time()
g <- read_graph(file = ncolName, format = "ncol", directed = T)
end_load <- proc.time()

time_load = end_load - start_load
cat("\n time_load : \n")
cat(time_load, "\n")


##### Calculate neighbourhood of each vertex #####
start_pre <- proc.time()
vertices <- V(g)
neighbourhood <- ego(g, order = depth, nodes = vertices, mode = "all", mindist = 1) 
names(neighbourhood) <- vertices$name
k <- ego_size(g, order = depth, nodes = vertices, mode = "all", mindist = 1) 
end_pre <- proc.time()

time_pre = end_pre - start_pre
cat("\n time_pre : \n")
cat(time_pre, "\n\n")


##### Normalize #####
start_norm <- proc.time()

df_features <- read_csv(featuresFile, show_col_types = FALSE)

newdf <- data.frame(matrix(NA, nrow=length(df_features$node), ncol=length(df_features)))
colnames(newdf) <- colnames(df_features)
newdf$node <- df_features$node
newdf$label <- df_features$label

for (feature in colnames(df_features)[2:8]) {
    cat(" ", feature) 
    featureTarget <- df_features[, c("node",feature)]
    time_feature <- system.time(newdf[, feature] <- sapply(seq(1,length(vertices)), function(x) normalizeFeature(featureTarget, neighbourhood, k, x)))
    cat(" - time: ", time_feature, "\n")
}

end_norm <- proc.time()

time_norm = end_norm - start_norm
cat("\n time_norm : \n")
cat(time_norm, "\n")



##### Store the new DataFrame in csv file #####
start_store <- proc.time()
write_csv(newdf, csvName) 
end_store <- proc.time()

time_store = end_store - start_store
cat("\n time_store : \n")
cat(time_store, "\n")

