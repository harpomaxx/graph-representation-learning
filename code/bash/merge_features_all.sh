#!/bin/bash
R_exe="Rscript"
merge_script_name="code/R/scripts/merge_features_cmd.R"
label_script_name="code/R/scripts/label_dataset_cmd.R"
merge_cmd_exe="$R_exe $merge_script_name"
label_cmd_exe="$R_exe $label_script_name"

usage() { echo "$0: [-f <featuresdir>] [-b <bcdir>]  [-o <outputdir>] " 1>&2; exit 1; }
while getopts ":f:o:l:b:" arg; do
    case "${arg}" in
        f)
            featuresdir=${OPTARG}
            ;;
        o)
            outputdir=${OPTARG}
            ;;
        b)
            bcdir=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))
# cleaning lock files
rm -f merged*end

if [ "$featuresdir" == "" ] || [ "$bcdir" == "" ] || [ "$outputdir" == "" ] ;then 
  echo "[] Parameters missing. Please use --h for looking at available parameters."  
  exit 
else
  echo "[bash] featuresdir: " $featuresdir
  echo "[bash] bcdir     :  " $bcdir
  echo "[bash] outputdir :  " $outputdir
fi
features_files=`find $featuresdir -name "*positive*.features"`


echo "[bash] Merging Features"
for feature_file in `echo $features_files`
do
	echo "[bash] Merging features for $feature_file"
	echo $merge_cmd_exe --allfeaturesfile $feature_file --bcfeaturefile $bcdir`basename $feature_file .features`.bc --output $outputdir`basename $feature_file .features`.csv 
	$merge_cmd_exe --allfeaturesfile $feature_file --bcfeaturefile $bcdir`basename $feature_file .features`.bc --output $outputdir`basename $feature_file .features`.csv 
done

## wait until all scripts have finished

echo "[bash] Checking if all processes finished"
finish=`find .  -name "merged*end"|wc -l`
filenum=`echo $features_files |wc -w`

if [ $finish != $filenum ];then
  echo "[bash] WARNING: Some scripts has not finished OK"
else
  echo "[bash] All the scripts has  finished OK"
fi
rm -f merged*end