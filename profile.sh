#!/bin/bash
make PG=1
./giraffe bench
gprof giraffe|prof2dot.py -s|dot -Tpng -o output.png
mv output.png /var/www/html/profile.png
echo "Saved to http://gc.matthewlai.ca/profile.png"
