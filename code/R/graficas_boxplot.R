library(ggplot2)

args <- commandArgs(TRUE)
cap <- paste(args[1], "_features_normalized.csv", sep = "")

featuresNorm <- readr::read_csv(cap, show_col_types=FALSE)

featuresNormGraficar <- data.table::melt(data.table::setDT(featuresNorm[,c(-1,-9)]), measure.vars=colnames(featuresNorm)[c(-1,-9)], variable.name="features", value.name="values")

p <- ggplot(featuresNormGraficar, aes(features,values)) + geom_boxplot() + stat_summary(fun="mean",geom="point",color="red")

ggsave(paste(args[1], "_boxplot.png", sep=""))
ggsave(paste(args[1], "_boxplot.pdf", sep=""))

