#!/bin/sh


for training_size in 10 50 100 500 1000 5000 10000 50000 100000
do
    for seed in 1 2 3 4 5
    do
	for participant_size in 5 6 7 8 9 10 12 14 16 18 20 25 30 35 40
	do
	    ./simulator.py --tag "Participant size vs training size" \
			   --number-of-training-experiments=$training_size \
			   --number-of-testing-experiments=1000 \
			   --experimental-data-rng-seed=$seed \
			   --control-group-size $participant_size \
			   --experiment-group-size $participant_size
	done
    done
done





