#!/bin/bash 
 
while getopts l:b: option 
do 
 case "${option}" 
 in 
 l) LR=${OPTARG};; 
 b) BS=${OPTARG};;
 esac 
done 

for lr in $LR
do
 for bs in $BS
 do
  python step_1_stl.py $bs $lr && python step_2_stl.py $bs $lr
 done
done
