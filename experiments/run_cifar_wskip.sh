#!/bin/bash

# Run CIFAR10 experiment on Skip-WGANomaly

declare -a arr=("airplane" "automobile" "bird" "cat" "deer" "dog" "frog" "horse" "ship" "truck" )
for i in "${arr[@]}";
do
    echo "Running CIFAR. Anomaly Class: $i "
    python train.py --dataset cifar10 --isize 32 --niter 15 --abnormal_class $i --model w_skipganomaly --display
done
exit 0
