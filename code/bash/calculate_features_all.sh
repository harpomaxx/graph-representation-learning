#!/bin/bash
R_exe="Rscript"
script_name="code/R/scripts/calculate_features_cmd.R"
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

ncol_files=`find $inputdir -name "*positive*.ncol"`
echo "[bash] Calculating Features"
for ncol in `echo $ncol_files`
do
	echo "[bash] calculating features for $ncol"
	echo $cmd_exe --input $ncol --output $outputdir`basename $ncol .ncol`.features
	$cmd_exe --input $ncol --output $outputdir`basename $ncol .ncol`.features &>features_`basename $ncol`.log &

done

## wait until all scripts have finished

echo "[bash] Waiting until all processes finished"
finish=`find .  -name "features*end"|wc -l`
filenum=`echo $ncol_files |wc -w`

while [ $finish != $filenum ]
do
  sleep 1
  finish=`find .  -name "features*end"|wc -l`
done
echo "[bash] Done"
rm -f features*end
rm -f features*log
