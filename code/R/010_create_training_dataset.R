# Create a dataframe with the normalized features of each capture (except 9, i.e. capture20110817) 


df <- data.frame(matrix(NA, ncol=10))
colnames(df) <- c('node','capture','ID','OD','IDW','ODW','BC','LCC','AC','label')


aux = c("20110810", "20110811", "20110812", "20110815", "20110815-2", "20110816", "20110816-2", "20110816-3", "20110817", "20110818", "20110818-2", "20110819", "20110815-3")


i = 1
for (date in aux) {
    if ( date != "20110817") {
        cap = readr::read_csv(paste("capture",date,"_features_normalized.csv", sep = ""),show_col_types = FALSE)
        cap <- tibble::add_column(cap, capture = rep(i, length(cap$node)), .after = 1)
        df <- dplyr::bind_rows(df, cap)
        i <- i + 1
        # Lo mismo en python:
        #cap.insert(1, 'capture', [i]*len(cap.iloc[:,0]))
        #df = pd.concat([df,cap], ignore_index=True)
        #i += 1
    } else {
        i <- i + 1
    }
}
df <- df[-1,]
rownames(df) <- NULL
readr::write_csv(df, "all_captures_except_9.csv")


# For kmeans 
X <- df[,3:9]
kmeans.re <- kmeans(X, centers = 4) #, nstart = 10)
print(kmeans.re$size)
##[1] 2870160    1507     508    2038

for(i in which(df$label=="botnet")){print(kmeans.re$cluster[i])}
## todas en cluster 1

