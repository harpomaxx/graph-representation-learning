#!/bin/bash
R_exe="Rscript"
script_name="code/R/scripts/calculate_ac_cmd.R"
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
  exit
else
echo "[bash] outputdir: " $outputdir
echo "[bash] intputdir: " $inputdir
fi

ncol_files=`find $inputdir -name "*positive*.ncol"`
#ncol_files=`dvc status calculate_bc|grep modified|awk -F: '{print $2}'|grep "ncol"`
echo "[bash] Calculating AC"
for ncol in `echo $ncol_files`
do
	echo "[bash] calculating ac for $ncol"
	echo $cmd_exe --input $ncol --output $outputdir/asis/`basename $ncol .ncol` --outputnorm $outputdir/normalized/`basename $ncol .ncol`
	$cmd_exe --input $ncol --output $outputdir/asis/`basename $ncol .ncol` --outputnorm $outputdir/normalized/`basename $ncol .ncol` >ac_`basename $ncol`.log &
	#$cmd_exe --input $ncol --output $outputdir`basename $ncol .ncol`.bc &>bc_`basename $ncol`.log &

done

## wait until all scripts have finished

echo "[bash] Waiting until all processes finished"
finish=`find .  -name "ac*end"|wc -l`
filenum=`echo $ncol_files |wc -w`
echo $filenum
while [ $finish != $filenum ]
do
  sleep 1
  finish=`find .  -name "ac*end"|wc -l`
  echo $finish
done
echo "[bash] Done"
rm -f ac*end
rm -f ac*log