#!/bin/Rscript

# Create a dataframe with the normalized features of each capture (except 9, i.e. capture20110817) 

# First capture
df <- readr::read_csv("capture20110810_features_normalized.csv", show_col_types = FALSE)
df <- tibble::add_column(df, capture = rep(1, length(df$node)), .after = 1)

# Captures in order from 2nd to 13th
aux = c("20110811", "20110812", "20110815", "20110815-2", "20110816", "20110816-2", "20110816-3", "20110817", "20110818", "20110818-2", "20110819", "20110815-3")

i = 2
for (date in aux) {
    if ( date != "20110817") {
        cap <- readr::read_csv(paste("capture", date, "_features_normalized.csv", sep = ""), show_col_types = FALSE)
        cap <- tibble::add_column(cap, capture = rep(i, length(cap$node)), .after = 1)
        df <- dplyr::bind_rows(df, cap)
        i <- i + 1
    } else {
        i <- i + 1
    }
}

readr::write_csv(df, "all_captures_except_9.csv")



