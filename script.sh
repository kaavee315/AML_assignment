#!/bin/bash

for ((i=32;i<=500;i*=2))
do
	for((j=4*i;j<=2000;j*=2))
	do
		echo "$i $j - " >>out_script
		python working_on.py $i $j 0.00005 >>out_script 
	done
done