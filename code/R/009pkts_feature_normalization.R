#!/bin/Rscript


suppressPackageStartupMessages(library(igraph))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(readr))


args <- commandArgs(TRUE)
ncolName <- paste(args[1], "_noZeroPkts.ncol", sep = "")
featuresFile <- paste("features/", args[1], "_features.csv", sep = "")
csvName <- paste(args[1], "_features_normalized.csv", sep = "")

cat("CAPTURE: ", args[1], "\n")
cat("Create graph from: ", ncolName, "\n")
cat("Store normalized features at: ", csvName, "\n")


normalizeFeature <- function(featureTarget, neighbourhood, k, index) {
    # Normalize feature
   
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
neighbourhood <- ego(g, order = 1, nodes = vertices, mode = "all", mindist = 1) 
names(neighbourhood) <- vertices$name
k <- ego_size(g, order = 1, nodes = vertices, mode = "all", mindist = 1) 
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
    tiempos <- system.time(newdf[, feature] <- sapply(seq(1,length(vertices)), function(x) normalizeFeature(featureTarget, neighbourhood, k, x)))
    cat(" - tiempo: ", tiempos, "\n")
}

end_norm <- proc.time()

time_norm = end_norm - start_norm
cat("\n time_norm : \n")
cat(time_norm, "\n")



##### Create a DataFrame and store it in csv file #####
start_store <- proc.time()
#df <- data.frame(AC) %>% as_tibble(rownames = "node")
write_csv(newdf, csvName) 
end_store <- proc.time()

time_store = end_store - start_store
cat("\n time_store : \n")
cat(time_store, "\n")
