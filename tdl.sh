#!/bin/bash

OMP_NUM_THREADS=16 nice -n 19 ./giraffe tdl training/ccrl4040_shuffled_5M.epd training/sts.epd &

giraffe_pid=$!

trap 'kill ${giraffe_pid}; exit' SIGINT

touch training.log

while true; do
	inotifywait -e modify -q -q training.log
	sleep 1
	python plot.py benchmark.log training.log
done
