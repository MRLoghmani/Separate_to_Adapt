#!/bin/bash 

DS="amazon webcam dslr"
DT="amazon webcam dslr"
num_knowns="10"
range_knowns="0-9"
num_exper="1 2 3"
gpu_id="0"

for e in $num_exper
do
  for s in $DS
  do
    for t in $DT
    do
      if [ $s != $t ]
      then
        echo $s
        echo $t
        python step_1_office31.py $s $t $num_knowns $range_knowns $e $gpu_id
        python step_2_office31.py $s $t $num_knowns $range_knowns $e $gpu_id
      fi
    done
  done
done


