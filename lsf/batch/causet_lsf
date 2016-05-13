#!/bin/bash

#BSUB -J CSET-%jobID%
#BSUB -o /gss_gpfs_scratch/cunningham/causet%jobID%/causet%jobID%.log
#BSUB -e /gss_gpfs_scratch/cunningham/causet%jobID%/causet%jobID%.err
#BSUB -q %queue%
#BSUB -n %totalcores%
#BSUB -R "span[ptile=%ncores%] select[hname!='%exclhost%']"
#BSUB -cwd /gss_gpfs_scratch/cunningham/causet%jobID%
#%waitfor%

###########################
#(C) Will Cunningham 2014 #
#         DK Lab          #
# Northeastern University #
###########################

# This is the LSF batch submission script for the CausalSet program
# It is intended to be submitted to LSF via the 'causet_bat' script

homedir=$CAUSET_HOME_DIR
work=/gss_gpfs_scratch/cunningham/causet%jobID%
usempi=%usempi%
if [[ %readjobid% -ne 0 && %readgraphid% -eq 0 ]] ; then
  usegraphlist=1
else
  usegraphlist=0
fi

# Setup peripheral directories and copy necessary files, as well as the binary
sed "s:@jobID@:%jobID%:g;s:@readjobid@:%readjobid%:g;s:@readgraphid@:%readgraphid%:g;s:@queue@:%queue%:g" < $homedir/lsf/setup/causet_setup | source /dev/stdin

# Do not edit below this

cd $work
tempfile1=hostlistrun
tempfile2=hostlist-tcp
echo $LSB_MCPU_HOSTS > $tempfile1
declare -a hosts
read -a hosts < ${tempfile1}
for ((i=0; i<${#hosts[@]}; i += 2)) ;
do
	HOST=${hosts[$i]}
	CORE=${hosts[(($i+1))]}
	echo $HOST 1 >> $tempfile2
done

# Do not edit above this

# Set OpenMP environment variables
export OMP_NUM_THREADS=%ncores%
export OMP_SCHEDULE="dynamic"
export OMP_NESTED=FALSE
export OMP_STACKSIZE="20M"

# Run nsamples jobs serially
if [ "$usegraphlist" -eq 1 ] ; then # Use a list of graph ids
  graphfile=graphlist
  readarray -t graphs < ${graphfile}
  for (( i=1; i<=${#graphs[@]}; i++ )) ; do
    echo "Reading Graph ${graphs[$i-1]}."
    if [ ${usempi} -eq 0 ] ; then # Do not use MPI
      ./CausalSet_%queue% %flags% --graph ${graphs[$i-1]} > CausalSet_Job-%jobID%_Sample-$i.log
    else # Do use MPI
      mpirun -np %mpinodes% -hostfile $tempfile2 -prot ./CausalSet_%queue% %flags% --graph %readgraphid% > CausalSet_Job-%jobID%_Sample-$i.log
    fi
    sleep 1
  done
else # Use a single graph id or create new graphs
  for (( i=1; i<=%nsamples%; i++ )); do
    echo "Starting Trial ${i} of %nsamples%"
    if [ ${usempi} -eq 0 ] ; then # Do not use MPI
      ./CausalSet_%queue% %flags% --graph %readgraphid% > CausalSet_Job-%jobID%_Sample-$i.log
    else # Do use MPI
      mpirun -np %mpinodes% -hostfile $tempfile2 -prot ./CausalSet_%queue% %flags% --graph %readgraphid% > CausalSet_Job-%jobID%_Sample-$i.log
    fi
    sleep 1
  done
fi

wait

# Do not edit below this

rm $work/$tempfile1
rm $work/$tempfile2

# Do not edit above this

if [ ${usegraphlist} -eq 1 ] ; then
  rm $work/$graphfile
fi
