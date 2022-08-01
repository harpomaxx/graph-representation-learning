#!/bin/Rscript

# 

df <- readr::read_csv("all_captures_except_9.csv", show_col_types = FALSE)

X <- df[,3:9]
numberClusters <- c(4,9,16,25,36,49,64,81,100,121,144,169,196,225)

botnets <- which(df$label == "botnet")
normals <- which(df$label == "normal")

totalObs <- length(df$node)
totalBots <- length(botnets)
totalHosts <- totalObs - totalBots

# k-means 
for (k in numberClusters) {
    sink(file = paste("kmeans_", k, "_clusters.txt", sep = ""), type = "output")
    
    vectAuxBotnet <- c()
    vectAuxNormal <- c()
    BOB <- 0
    HOB <- 0
    
    cat("\n For k = ", k, ": \n")
    time_kmeans <- system.time(kmeans.re <- kmeans(X, centers = k, nstart = 10))
    
    aux <- as.data.frame(kmeans.re$cluster)
    colnames(aux) <- paste("kmeans_", as.character(k), sep = "")

    df <- dplyr::bind_cols(df, aux)
    
    benignCluster <- which.max(kmeans.re$size)
    
    for (i in botnets) {
        vectAuxBotnet <- append(vectAuxBotnet, kmeans.re$cluster[i])
        if (kmeans.re$cluster[i] != benignCluster) {
            BOB <- BOB + 1
        } 
    }
    for (i in normals) {
        vectAuxNormal <- append(vectAuxNormal, kmeans.re$cluster[i])
    }
    HOB <- sum(kmeans.re$size)-kmeans.re$size[benignCluster]
    
    cat("\nwhich cluster each botnet belongs to: \n")
    print(vectAuxBotnet)
    cat("\nwhich cluster each normal host belongs to: \n")
    print(vectAuxNormal)
    
    cat("\ncenters: \n")
    print(kmeans.re$centers)
    cat("\nsize of clusters: \n")
    print(kmeans.re$size)
    
    cat("\nHOB = ", HOB, "\n")
    cat("HOB% = ", (HOB/totalHosts)*100, "\n")
    cat("BOB = ", BOB, "\n")
    cat("BOB% = ", (BOB/totalBots)*100, "\n")

    cat("\ntime: \n")
    print(time_kmeans)
    sink()
}

readr::write_csv(df, "features_normalized_and_kmeans.csv") 

