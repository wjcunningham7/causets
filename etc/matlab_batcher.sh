#!/bin/sh

matlab_exec=/opt/MathWorks/Matlab/R2014a/bin/matlab
#X="${1}(${2})"
#echo ${X} > matlab_command_${2}.m
#cat matlab_command_${2}.m
#${matlab_exec} -nojvm -nodisplay -nosplash < matlab_command_${2}.m
#rm matlab_command_${2}.m

${matlab_exec} -nojvm -nodisplay -nosplash -r ${1}
