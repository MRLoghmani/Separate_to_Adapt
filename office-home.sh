#!/bin/bash 

DS="art clipart product real_world"
DT="art clipart product real_world"
num_knowns="25"
range_knowns="0-24"
num_exper="2 3"
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
        python step_1_officehome.py $s $t $num_knowns $range_knowns $e $gpu_id
        python step_2_officehome.py $s $t $num_knowns $range_knowns $e $gpu_id
      fi
    done
  done
done

