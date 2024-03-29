#!/bin/Rscript

library(unix)
rlimit_as(1e12)

#args <- commandArgs(TRUE)
#TIPO <- args[1]

df <- readr::read_csv("all_captures_except_9.csv", show_col_types = FALSE)

X <- df[,3:9]
numberClusters <- c(4,9,16,25,36,49,64,81,100,121,144,169,196,225)
#c(4,16,36,64,100,144,196)
#c(4,9,16,25,36,49,64,81,100,121,144,169,196,225)

botnets <- which(df$label == "botnet")
normals <- which(df$label == "normal")

totalObs <- length(df$node)
totalBots <- length(botnets)
totalHosts <- totalObs - totalBots

k <- c()
HOB <- c()
HOB_percent <- c()
BOB <- c()
BOB_percent <- c()


kmedoids_pam <- function(df, k) {
  # Code extracted from:
  # https://towardsdatascience.com/a-deep-dive-into-partitioning-around-medoids-a77d9b888881
  
  # define manhattan distance
  manhattan_distance <- function(x, y){
    return(sum(abs(x - y)))
  }
  
  # Calculate distances between all points and all current medoids
  calculate_distances <- function(df, medoids) {
    distances <- matrix(NA, nrow = nrow(df), ncol = nrow(medoids))
    for (object_id in 1:nrow(df)) {
      for (medoid_id in 1:nrow(medoids)) {
        distances[object_id, medoid_id] <- manhattan_distance(df[object_id, ], medoids[medoid_id, ])
      }
    }
    return(distances)
  }
  
  # Calculate the total cost 
  calculate_cost <- function(df, medoids) {
    distances <- calculate_distances(df, medoids)
    costs <- rep(NA, ncol(distances))
    cluster_id <- apply(distances, 1, which.min)
    
    # number of columns in the distance matrix equals the number of clusters
    for (cluster in 1:ncol(distances)) {
      costs[cluster] <- sum(distances[cluster_id == cluster, cluster])
    }
    cost <- sum(costs)
    return(cost)
  }
  
  # Get non medoids. This function is slightly different to the one of the Lloyd
  # style kmedoids, because this time we are comparing against multiple medoids.
  # The concept is: go through the data frame by row (first apply call)
  # subtract the current row from the medoids data frame
  # Go through the resulting differences. If one of them is all 0 then the current
  # row is a medoid. 
  # We cannot use rowSums for this, because a row with entries c(-1, 1) would
  # also sum to 0, but is not a medoid!
  get_non_medoids <- function(df, medoids) {
    non_medoids <- !apply(df, 1, function(x) {
      differences <- sweep(medoids, 2, x)
      is_medoid <- any(apply(differences, 1, function(y) all(y == 0)))
      is_medoid
    })
    non_medoids <- df[non_medoids, ]
    return(non_medoids)
  }
  
  # Get the best medoid for a potential swap
  get_best_swap_medoid <- function(df, medoid) {
    best_cost <- Inf
    best_medoid <- medoid
    non_medoids <- get_non_medoids(df, medoid)
    
    for (non_medoid_id in 1:nrow(non_medoids)) {
      candidate_medoid <- non_medoids[non_medoid_id, ]
      this_cost <- calculate_cost(df, candidate_medoid)
      if (this_cost < best_cost) {
        best_cost <- this_cost
        best_medoid <- candidate_medoid
      }
    }
    out <- cbind(cost, best_medoid)
    return(out)
  }
  
  # BUILD phase
  # Select first medoid as the one which has the smallest cost
  distances <- as.matrix(dist(df, method = "manhattan"))
  distances <- colSums(distances)
  medoid_id <- which.min(distances) # In case of ties this will return the first minimum
  medoids <- df[medoid_id, ]
  
  # From the remaining non_medoids select the next one that has the smallest cost
  # until we have k medoids
  while (nrow(medoids) < k) {
    non_medoids <- get_non_medoids(df, medoids)
    best_cost <- Inf
    for (non_medoid_id in 1:nrow(non_medoids)) {
      candidate_medoid <- non_medoids[non_medoid_id, ]
      candidate_cost <- calculate_cost(df, rbind(medoids, candidate_medoid))
      if (candidate_cost < best_cost) {
        best_medoid <- candidate_medoid
        best_cost <- candidate_cost
      }
    }
    
    # Add the best medoid to the medoids
    medoids <- rbind(medoids, best_medoid)
  }
  
  # Calculate initial cost  
  cost <- calculate_cost(df, medoids)
  
  # SWAP phase
  # Run until algorithm converged
  iteration <- 0 # To keep track how many iterations we needed.
  
  while (TRUE) {
    # In contrast to Lloyd style k-medoids, consider the complete data set
    # for potential swaps, not only the current cluster of the medoid
    candidate_swaps <- list()
    for (medoid_id in 1:nrow(medoids)) {
      this_medoid <- medoids[medoid_id, ]
      candidate_swaps[[medoid_id]] <- get_best_swap_medoid(df, this_medoid)
    }
    candidate_swaps <- Reduce(rbind, candidate_swaps)
    
    # Select and perform the best swap
    medoid_to_swap <- which.min(candidate_swaps$cost)
    medoids[medoid_to_swap, ] <- candidate_swaps[medoid_to_swap, colnames(candidate_swaps) != "cost", drop = FALSE]
    rownames(medoids)[medoid_to_swap] <- rownames(candidate_swaps)[medoid_to_swap] 
    
    new_cost <- calculate_cost(df, medoids)
    
    if (new_cost < cost) {
      cost <- new_cost
      iteration <- iteration + 1
    } else {
      # If cost no longer decreases break out of the loop and return results
      print(paste("Converged after", iteration, "iterations."))
      distances <- calculate_distances(df, medoids)
      cluster_id <- apply(distances, 1, which.min)
      out <- list(cluster_id = cluster_id,
                  medoids = medoids)
      return(out)
    }
  }
  
}


# k-medoids
for (cluster in numberClusters) {
    print(cluster)
    k <- append(k, cluster)
    zz <- file(paste("kmedoids_pam_", cluster, "_clusters.txt", sep = ""), open = "wt")
    sink(zz)
    sink(zz, type = "message")

    #sink(file = paste("kmeans_", cluster, "_clusters_",TIPO,".txt", sep = ""), type = "output")
    
    vectAuxBotnet <- c()
    vectAuxNormal <- c()
    BOB_val <- 0
    HOB_val <- 0
    
    cat("\n For k = ", cluster, ": \n")
    #gc()
    time_kmeans <- system.time(kmedoids <- kmedoids_pam(X, cluster))
    
    aux <- as.data.frame(kmeans.re$cluster)
    colnames(aux) <- paste("kmeans_", as.character(cluster), sep = "")

    df <- dplyr::bind_cols(df, aux)
    
    benignCluster <- which.max(kmeans.re$size)
    
    for (i in botnets) {
        vectAuxBotnet <- append(vectAuxBotnet, kmeans.re$cluster[i])
        if (kmeans.re$cluster[i] != benignCluster) {
            BOB_val <- BOB_val + 1
        } 
    }
    BOB <- append(BOB, BOB_val)
    BOB_percent_val <- (BOB_val/totalBots)*100
    BOB_percent <- append(BOB_percent, BOB_percent_val)

    for (j in normals) {
        vectAuxNormal <- append(vectAuxNormal, kmeans.re$cluster[j])
    }
    HOB_val <- totalHosts - kmeans.re$size[benignCluster]
    HOB <- append(HOB, HOB_val)
    HOB_percent_val <- (HOB_val/totalHosts)*100
    HOB_percent <- append(HOB_percent, HOB_percent_val)

    cat("\nwhich cluster each botnet belongs to: \n")
    print(vectAuxBotnet)
    cat("\nwhich cluster each normal host belongs to: \n")
    print(vectAuxNormal)
    
    cat("\ncenters: \n")
    print(kmeans.re$centers)
    cat("\nsize of clusters: \n")
    print(kmeans.re$size)
    
    cat("\nHOB = ", HOB_val, "\n")
    cat("HOB% = ", HOB_percent_val, "\n")
    cat("BOB = ", BOB_val, "\n")
    cat("BOB% = ", BOB_percent_val, "\n")

    cat("\ntime: \n")
    print(time_kmeans)
    #sink()
    sink(type = "message")
    sink()

    print(warnings())
    cat("\n\n")
    rm(kmeans.re)
    gc()
}

readr::write_csv(df, paste("features_normalized_and_kmeans_",TIPO,".csv",sep="")) 

hobob <- data.frame(k,HOB,HOB_percent,BOB,BOB_percent)
readr::write_csv(hobob, paste("HOB_BOB_table_",TIPO,".csv",sep=""))
