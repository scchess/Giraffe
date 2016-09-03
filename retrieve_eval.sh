#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "Usage: $0 <file>"
    exit
fi

scp gc:/home/matthewlai/giraffe/$1 .
