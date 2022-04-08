#!/bin/Rscript

suppressPackageStartupMessages(library(igraph))
suppressPackageStartupMessages(library(dplyr))

args <- commandArgs(TRUE)

ncolName <- paste(args[1], "_noZeroB.ncol", sep = "")
csvName <- paste(args[1], "_features.csv", sep = "")

which_capture <- args[1]
path_d_dw <- paste(args[2], which_capture, "_D_DW.csv", sep = "")
path_bc <- paste(args[3], which_capture, "_BC.csv", sep = "")
path_lcc <- paste(args[4], which_capture, "_LCC.csv", sep = "")
path_ac <- paste(args[5], which_capture, "_AC.csv", sep = "")


sprintf("CAPTURE: %s", which_capture)
sprintf("Create graph from: %s", ncolName)
sprintf("Store features at: %s", csvName)


labeling <- function(capture, node) {
    # label each node as specified in https://www.stratosphereips.org/datasets-ctu13
     
    if (node == "147.32.84.165") {
        label <- "botnet"
    } else if ( node == "147.32.84.170" || node == "147.32.84.164" || node == "147.32.87.36" || node == "147.32.80.9" ) {
        label <- "normal"
    } else if ( (capture != "capture20110811" && capture != "capture20110816-2") && (node == "147.32.84.134" || node == "147.32.87.11") ) {
        label <- "normal"
    } else if ( capture == "capture20110811" && node == "147.32.87.11" ) {
        label <- "normal"
    } else if ( capture == "capture20110816-2" && node == "147.32.84.134" ) {
        label <- "normal"
    } else if ( (capture == "capture20110817" || capture == "capture20110818" || capture == "capture20110818-2" || capture == "capture20110819") && (node == "147.32.84.191" || node == "147.32.84.192") ) {
        label <- "botnet"
    } else if ( (capture == "capture20110817" || capture == "capture20110818") && (node == "147.32.84.193" || node == "147.32.84.204" || node == "147.32.84.205" || node == "147.32.84.206" || node == "147.32.84.207" || node == "147.32.84.208" || node == "147.32.84.209") ) {
        label <- "botnet"
    } else {
        label <- "background"
    }
    
    return(label)
}

##### Load graph and features #####
g <- read_graph(file = ncolName, format = "ncol", directed = T)
d_dw <- read.csv(path_d_dw)
bc <- read.csv(path_bc)
lcc <- read.csv(path_lcc)
ac <- read.csv(path_ac)

g_vertices <- V(g)
verticesNames <- names(g_vertices)
tmp <- lapply(verticesNames, function(x) labeling(which_capture,x))

names(tmp) <- verticesNames
labels <- unlist(tmp) %>% data.frame(label = .) %>% as_tibble(rownames = "node")


df <- left_join(d_dw, bc, by = "node") %>% left_join(., lcc, by = "node") %>% left_join(., ac, by = "node") %>% left_join(., labels, by = "node")
write.csv(df, csvName, row.names = FALSE)

