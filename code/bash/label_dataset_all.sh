#!/bin/bash
# script for labeling CTU13 node dataset
R_exe="Rscript"
label_script_name="code/R/scripts/label_dataset_cmd.R"
label_cmd_exe="$R_exe $label_script_name"

usage() { echo "$0: [-i <inputdir>] [-l <labelsdir>] [-o <outputdir>] " 1>&2; exit 1; }
while getopts ":i:o:l:" arg; do
    case "${arg}" in
        i)
            inputdir=${OPTARG}
            ;;
        o)
            outputdir=${OPTARG}
            ;;
        l)
            labelsdir=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))
# cleaning lock files
rm -f labeled*end

if [ "$inputdir" == "" ] || [ "$outputdir" == "" ] || [ "$labelsdir" == "" ] ;then 
  echo "[] Parameters missing. Please use --h for looking at available parameters."  
  exit 
else
  echo "[bash] inputdir  :  " $inputdir
  echo "[bash] outputdir :  " $outputdir
  echo "[bash] labelsdir :  " $labelsdir
  echo "[bash] outputdir :  " $outputdir
fi
dataset_files=`find $inputdir -name "*positive*.csv"`

echo "[bash] Labeling Datasets"
for dataset_file in `echo $dataset_files`
do
	echo "[bash] Labeling dataset for $dataset_file"
	label_file=`echo $dataset_file|sed -s 's/.*\(cap.*\)bine.*/\1nodes.labels/g'`
	echo $label_cmd_exe --input $dataset_file --labelsfile $labelsdir$label_file --output $outputdir`basename $dataset_file .csv`.labeled.csv 
	$label_cmd_exe --input $dataset_file --labelsfile $labelsdir$label_file --output $outputdir`basename $dataset_file .csv`.labeled.csv 
done

## wait until all scripts have finished

echo "[bash] Checking if all processes finished"
finish=`find .  -name "labeled*end"|wc -l`
filenum=`echo $dataset_files |wc -w`

if [ $finish != $filenum ];then
  echo "[bash] WARNING: Some scripts has not finished OK"
else
  echo "[bash] All the scripts has  finished OK"
fi
rm -f labeled*end