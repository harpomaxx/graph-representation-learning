#!/bin/bash
# script normalizing features
R_exe="Rscript"
normalize_script_name="code/R/scripts/normalize_features_cmd.R"
normalize_cmd_exe="$R_exe $normalize_script_name"

usage() { echo "$0: [-n <ncoldir>] [-f <featuresdir>] [-o <outputdir>] " 1>&2; exit 1; }
while getopts ":n:o:f:" arg; do
    case "${arg}" in
        n)
            ncoldir=${OPTARG}
            ;;
        o)
            outputdir=${OPTARG}
            ;;
        f)
            featuresdir=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))
# cleaning lock files
rm -f normalizeed*end

if [ "$ncoldir" == "" ] || [ "$outputdir" == "" ] || [ "$featuresdir" == "" ] ;then 
  echo "[] Parameters missing. Please use --h for looking at available parameters."  
  exit 
else
  echo "[bash] inputdir      :    " $ncoldir
  echo "[bash] outputdir     :    " $outputdir
  echo "[bash] featuresdirs  :    " $featuresdir
fi
graph_files=`find $ncoldir -name "*positive*.ncol"`

echo "[bash] Normalizing features"
for graph_file in `echo $graph_files`
do
	echo "[bash] Normalizing features for $graph_file"
	#label_file=`echo $dataset_file|sed -s 's/.*\(cap.*\)bine.*/\1nodes.labels/g'`
	$normalize_cmd_exe --ncolfile $graph_file --featuresfile $featuresdir`basename $graph_file .ncol`.csv  --output $outputdir`basename $graph_file .csv`.normalized.csv 
done

## wait until all scripts have finished

echo "[bash] Checking if all processes finished"
finish=`find .  -name "fnormalize*end"|wc -l`
filenum=`echo $graph_files |wc -w`

if [ $finish != $filenum ];then
  echo "[bash] WARNING: Some scripts has not finished OK"
else
  echo "[bash] All the scripts has  finished OK"
fi
rm -f fnormalize*end