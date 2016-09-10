#!/bin/bash

OMP_NUM_THREADS=20 nice -n 19 ./giraffe tdl training/ccrl4040_shuffled_5M.epd
