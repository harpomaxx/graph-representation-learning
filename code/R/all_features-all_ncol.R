#!/bin/Rscript

# Create a dataframe with features of each capture (except 9, i.e. capture20110817) 

# Note: pwd == /home/tati/Nextcloud/BotChase/graph-representation-learning/rawdata/harpo

# First capture
df_features <- readr::read_csv("captures-labeled-10-10-2022/capture20110810.binetflow.labels-positive-weights.labeled.csv", show_col_types = FALSE)
df_features$node <- sub("^", "1-", df_features$node)

df_ncol <- readr::read_delim("ncol/capture20110810.binetflow.labels-positive-weights.ncol", delim=" ", col_names=c("origen","destino","peso"), show_col_types = FALSE)
df_ncol$origen <- sub("^", "1-", df_ncol$origen)
df_ncol$destino <- sub("^", "1-", df_ncol$destino)

# Captures in order from 2nd to 13th
aux = c("20110811", "20110812", "20110815", "20110815-2", "20110816", "20110816-2", "20110816-3", "20110817", "20110818", "20110818-2", "20110819", "20110815-3")

i = 2
for (date in aux) {
    if ( date != "20110817") {
        cap_features <- readr::read_csv(paste("captures-labeled-10-10-2022/capture", date, ".binetflow.labels-positive-weights.labeled.csv", sep = ""), show_col_types = FALSE)
        cap_features$node <- sub("^", paste(as.character(i),"-",sep=""), cap_features$node)
        df_features <- dplyr::bind_rows(df_features, cap_features)
        
        cap_ncol <- readr::read_delim(paste("ncol/capture", date, ".binetflow.labels-positive-weights.ncol", sep = ""), delim=" ", col_names=c("origen","destino","peso"), show_col_types = FALSE)
        cap_ncol$origen <- sub("^", paste(as.character(i),"-",sep=""), cap_ncol$origen)
        cap_ncol$destino <- sub("^", paste(as.character(i),"-",sep=""), cap_ncol$destino)
        df_ncol <- dplyr::bind_rows(df_ncol, cap_ncol)
        i <- i + 1
    } else {
        i <- i + 1
    }
}

readr::write_csv(df_features, "all_captures_except_9_FEATURES.pkts.SINnorm.csv")

readr::write_delim(df_ncol, "all_captures_except_9_GRAFOS.pkts.ncol", col_names=FALSE)

###########################################################################################

# First capture training
df_features <- readr::read_csv("captures-labeled-10-10-2022/capture20110810.binetflow.labels-positive-weights.labeled.csv", show_col_types = FALSE)
df_features$node <- sub("^", "1-", df_features$node)

df_ncol <- readr::read_delim("ncol/capture20110810.binetflow.labels-positive-weights.ncol", delim=" ", col_names=c("origen","destino","peso"), show_col_types = FALSE)
df_ncol$origen <- sub("^", "1-", df_ncol$origen)
df_ncol$destino <- sub("^", "1-", df_ncol$destino)


# Captures in order from 2nd to 13th
aux = c("20110811", "20110812", "20110815", "20110815-2", "20110816", "20110816-2", "20110816-3", "20110817", "20110818", "20110818-2", "20110819", "20110815-3")

i = 2
for (date in aux) {
    if ( (date != "20110816-3") & (date != "20110817") & (date != "20110819") ) {
        cap_features <- readr::read_csv(paste("captures-labeled-10-10-2022/capture", date, ".binetflow.labels-positive-weights.labeled.csv", sep = ""), show_col_types = FALSE)
        cap_features$node <- sub("^", paste(as.character(i),"-",sep=""), cap_features$node)
        df_features <- dplyr::bind_rows(df_features, cap_features)
        
        cap_ncol <- readr::read_delim(paste("ncol/capture", date, ".binetflow.labels-positive-weights.ncol", sep = ""), delim=" ", col_names=c("origen","destino","peso"), show_col_types = FALSE)
        cap_ncol$origen <- sub("^", paste(as.character(i),"-",sep=""), cap_ncol$origen)
        cap_ncol$destino <- sub("^", paste(as.character(i),"-",sep=""), cap_ncol$destino)
        df_ncol <- dplyr::bind_rows(df_ncol, cap_ncol)
        i <- i + 1
    } else {
        i <- i + 1
    }
}

readr::write_csv(df_features, "training_FEATURES.pkts.SINnorm.csv")
readr::write_delim(df_ncol, "training_GRAFOS.pkts.ncol", col_names=FALSE)

#########################

# Validation

df_featuresVal <- readr::read_csv("captures-labeled-10-10-2022/capture20110816-3.binetflow.labels-positive-weights.labeled.csv", show_col_types = FALSE)
df_featuresVal$node <- sub("^", "8-", df_featuresVal$node)

df_ncolVal <- readr::read_delim("ncol/capture20110816-3.binetflow.labels-positive-weights.ncol", delim=" ", col_names=c("origen","destino","peso"), show_col_types = FALSE)
df_ncolVal$origen <- sub("^", "8-", df_ncolVal$origen)
df_ncolVal$destino <- sub("^", "8-", df_ncolVal$destino)

cap_featuresVal <- readr::read_csv("captures-labeled-10-10-2022/capture20110819.binetflow.labels-positive-weights.labeled.csv", show_col_types = FALSE)
cap_featuresVal$node <- sub("^", "12-", cap_featuresVal$node)
df_featuresVal <- dplyr::bind_rows(df_featuresVal, cap_featuresVal)
        
cap_ncolVal <- readr::read_delim("ncol/capture20110819.binetflow.labels-positive-weights.ncol", delim=" ", col_names=c("origen","destino","peso"), show_col_types = FALSE)
cap_ncolVal$origen <- sub("^", "12-", cap_ncolVal$origen)
cap_ncolVal$destino <- sub("^", "12-", cap_ncolVal$destino)
df_ncolVal <- dplyr::bind_rows(df_ncolVal, cap_ncolVal)

readr::write_csv(df_featuresVal, "validation_FEATURES.pkts.SINnorm.csv")
readr::write_delim(df_ncolVal, "validation_GRAFOS.pkts.ncol", col_names=FALSE)
