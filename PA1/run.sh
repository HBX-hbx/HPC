#!/bin/bash

# run on 1 machine * 28 process, feel free to change it!
if [ "$2" -le 1000 ]; then
  srun -N 1 -n 1 --cpu-bind=cores $*
elif [ "$2" -le 10000 ]; then
  srun -N 1 -n 10 --cpu-bind=cores $*
elif [ "$2" -le 100000 ]; then
  srun -N 1 -n 20 --cpu-bind=cores $*
else
  srun -N 2 -n 40 --cpu-bind=cores $*
fi

# srun -N 1 -n 2 ./odd_even_sort 256 ./my_data/256.dat

