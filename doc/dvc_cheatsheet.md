

```
 dvc run -n convert -d rawdata/ctu-13/capture20110810.binetflow.labels.gz -d code/R/convert_net2graph.R  --outs-persist  data/pajek   Rscript code/R/convert_net2graph.R --input rawdata/ctu-13/capture20110810.binetflow.labels.gz --output data/pajek/capture20110810.binetflow.pajek

```
