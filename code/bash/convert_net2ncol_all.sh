#!/bin/bash
R_exe="Rscript"
script_name="code/R/scripts/convert_net2ncol_cmd.R"
cmd_exe="$R_exe $script_name"
outputdir="data/ncol/"

netflow_files=" 
rawdata/ctu-13/capture20110810.binetflow.labels.gz
rawdata/ctu-13/capture20110811.binetflow.labels.gz
rawdata/ctu-13/capture20110812.binetflow.labels.gz
rawdata/ctu-13/capture20110815-2.binetflow.labels.gz
rawdata/ctu-13/capture20110815-3.binetflow.labels.gz
rawdata/ctu-13/capture20110815.binetflow.labels.gz
rawdata/ctu-13/capture20110816-2.binetflow.labels.gz
rawdata/ctu-13/capture20110816-3.binetflow.labels.gz
rawdata/ctu-13/capture20110816.binetflow.labels.gz
rawdata/ctu-13/capture20110817.binetflow.labels.gz
rawdata/ctu-13/capture20110818-2.binetflow.labels.gz
rawdata/ctu-13/capture20110818.binetflow.labels.gz
rawdata/ctu-13/capture20110819.binetflow.labels.gz "

echo "[bash] converting to ncol"
for netflow in `echo $netflow_files`
do
	echo "[bash[] converting $netflow"
	echo $cmd_exe --input $netflow --output $outputdir`basename $netflow .gz`.ncol
	$cmd_exe --input $netflow --output $outputdir`basename $netflow .gz`.ncol
done

