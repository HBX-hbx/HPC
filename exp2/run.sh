#!/bin/bash

for N in 1 2
do
	for n in 1024 32768 1048576
	do
		for p in 2 4 8 16
		do
			srun -N $N -n $p ./allreduce 10 $n
		done
	done
done

