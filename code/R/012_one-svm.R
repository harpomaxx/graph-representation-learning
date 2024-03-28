#!/bin/Rscript

library(e1071)

df_complete <- readr::read_csv("kmeans_HW_unix_gc/features_normalized_and_kmeans_Hartigan-Wong.csv", show_col_types = FALSE)
df_kmeans4 <- df_complete[,c(3:9,11)]

test <- readr::read_csv("capture20110817_features_normalized.csv", show_col_types = FALSE)

benignCluster <- 3

X <- dplyr::filter(df_kmeans4, df_kmeans4$kmeans_4 != benignCluster)[,1:(length(df_kmeans4)-1)]

#if (length(X$ID) != 4053) {
#    quit("no", status=1)
#} else {
    vectFactor <- as.factor(c("host", "botnet"))
    y <- rep(vectFactor[1], length(X$ID))
    model <- svm(X, y, type='one-classification', scale=FALSE)
    print(model)
    print(summary(model))

    pred <- predict(model, test[,2:8])

    readr::write_csv(dplyr::bind_cols(test,pred), "capture20110817_predict.csv")
    
    cat("Total obs: ", length(test$node), "\n")
    cat("Total normal: ", sum(pred, na.rm=TRUE), "\n")
    cat("Total botnets: ", length(test$node)-sum(pred, na.rm=TRUE), "\n")
    print(warnings())
#}


