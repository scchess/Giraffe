#!/bin/bash
make -j 16 PG=1
./giraffe bench
gprof giraffe|gprof2dot -s|dot -Tpng -o output.png
mv output.png /var/www/html/profile.png
echo "Saved to http://bigfoot/profile.png"
