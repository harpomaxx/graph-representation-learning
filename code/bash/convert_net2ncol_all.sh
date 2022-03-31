#!/bin/bash
R_exe="Rscript"
script_name="code/R/scripts/convert_net2ncol_cmd.R"
cmd_exe="$R_exe $script_name"
usage() { echo "$0: [-i <inputdir>] [-o <outputdir>]" 1>&2; exit 1; }
while getopts ":i:o:" arg; do
    case "${arg}" in
        i)
            inputdir=${OPTARG}
            ;;
        o)
            outputdir=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ "$outputdir" == "" ] || [ "$inputdir" == "" ] ;then 
  echo "[] Parameters missing. Please use --h for look at available parameters."
else
echo "[bash] outputdir: " $outputdir
echo "[bash] intputdir: " $inputdir
fi

netflow_files=`find $inputdir -name "*binetflow.labels.gz"`
#echo "[bash] cleaning metrics files"
#>metrics/generate_ncol.yaml

echo "[bash] converting to ncol"
for netflow in `echo $netflow_files`
do
	echo "[bash[] converting $netflow"
	echo $cmd_exe --input $netflow --output $outputdir`basename $netflow .gz`.ncol
	$cmd_exe --input $netflow --output $outputdir`basename $netflow .gz`.ncol
done

