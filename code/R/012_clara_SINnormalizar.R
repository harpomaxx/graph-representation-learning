#!/bin/Rscript

library(cluster)
library(unix)
rlimit_as(1e12)

args <- commandArgs(TRUE)
METRIC <- args[1]

df <- readr::read_csv("all_captures_except_9_SINnormalizar.csv", show_col_types = FALSE)

X <- df[,3:9]
numberClusters <- c(4,9,16,25,36,49,64,81,100,121,144,169,196,225)


botnets <- which(df$label == "botnet")
normals <- which(df$label == "normal")

totalObs <- length(df$node)
totalBots <- length(botnets)
totalHosts <- totalObs - totalBots
cat("\ntotalObs = ", totalObs, "\ntotalBots = ", totalBots, "\ntotalHosts = ", totalHosts, "\n")

k <- c()
HOB <- c()
HOB_percent <- c()
BOB <- c()
BOB_percent <- c()


# clara
for (cluster in numberClusters) {
    print(cluster)
    k <- append(k, cluster)
    zz <- file(paste("clara_", cluster, "_clusters_", METRIC, ".txt", sep = ""), open = "wt")
    sink(zz)
    sink(zz, type = "message")

    vectAuxBotnet <- c()
    vectAuxNormal <- c()
    BOB_val <- 0
    HOB_val <- 0

    cat("\n For k = ", cluster, ": \n")
    time_clara <- system.time(clara.res <- clara(X, k = cluster, metric = METRIC, samples = 300, pamLike = TRUE))
    print(clara.res$call)
    
    aux <- as.data.frame(clara.res$clustering)
    colnames(aux) <- paste("clara_", as.character(cluster), sep = "")

    df <- dplyr::bind_cols(df, aux)

    benignCluster <- which.max(clara.res$clusinfo[,1])

    for (i in botnets) {
        vectAuxBotnet <- append(vectAuxBotnet, clara.res$clustering[i])
        if (clara.res$clustering[i] != benignCluster) {
            BOB_val <- BOB_val + 1
        }
    }
    BOB <- append(BOB, BOB_val)
    BOB_percent_val <- (BOB_val/totalBots)*100
    BOB_percent <- append(BOB_percent, BOB_percent_val)

    for (j in normals) {
        vectAuxNormal <- append(vectAuxNormal, clara.res$clustering[j])
    }
    HOB_val <- totalObs - clara.res$clusinfo[benignCluster,1] - BOB_val
    HOB <- append(HOB, HOB_val)
    HOB_percent_val <- (HOB_val/totalHosts)*100
    HOB_percent <- append(HOB_percent, HOB_percent_val)

    cat("\nbenignCluster = ", benignCluster, "\n")
    cat("\nwhich cluster each botnet belongs to: \n")
    print(vectAuxBotnet)
    
    clusterHet <- c(vectAuxBotnet[1])
    auxSumaClusterHet <- c(sum(vectAuxBotnet == clusterHet[1]))
    for (m in seq(1,length(vectAuxBotnet))) {
        if (any(vectAuxBotnet[m] %in% clusterHet)) {
            next
        } else {
            clusterHet <- append(clusterHet, vectAuxBotnet[m]) 
            auxSumaClusterHet <- append(auxSumaClusterHet, sum(vectAuxBotnet == vectAuxBotnet[m]))
        }
    }
    sizeClusterHet <- c(clara.res$clusinfo[clusterHet[1],1])
    for (n in clusterHet[-1]) {
        sizeClusterHet <- append(sizeClusterHet, clara.res$clusinfo[n,1])
    }
    infoBotnets <- data.frame(HetClust=clusterHet, botnets_in_HetClust=auxSumaClusterHet, size_HetClust=sizeClusterHet)
    cat("\ninfo about botnets: \n   HetClust: heterogeneous clusters \n   botnets_in_HetClust: how many botnets are there? \n   sizeHetClust: how many nodes (hosts and bots) are there? \n\n")
    print(infoBotnets)

    cat("\nwhich cluster each normal host belongs to: \n")
    print(vectAuxNormal)

    cat("\ncenters: \n")
    print(clara.res$medoids)
    cat("\nsize, max_diss, av_diss, isolation of clusters: \n")
    print(clara.res$clusinfo)

    cat("\nHOB = ", HOB_val, "\n")
    cat("HOB% = ", HOB_percent_val, "\n")
    cat("BOB = ", BOB_val, "\n")
    cat("BOB% = ", BOB_percent_val, "\n")

    cat("\ntime: \n")
    print(time_clara)
    #sink()
    sink(type = "message")
    sink()

    print(warnings())
    cat("\n\n")
    rm(clara.res)
    gc()
}

readr::write_csv(df, paste("features_and_clara_",METRIC,".csv",sep=""))

hobob <- data.frame(k,HOB,HOB_percent,BOB,BOB_percent)
readr::write_csv(hobob, paste("HOB_BOB_table_",METRIC,".csv",sep=""))


