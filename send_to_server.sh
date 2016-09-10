#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "Usage: $0 version_name"
	exit 1
fi

VER=$1

TARGET="/home/matthewlai/playground/players/$VER"

ssh gc mkdir $TARGET

rsync -avz --exclude '.hg' --include 'sts.epd' --exclude '*.epd' --exclude '*.epd.gz' --exclude '*.log' --exclude '*.o' --exclude '*.a' --exclude '*.d' --exclude 'giraffe' . gc:$TARGET/

echo $VER > version_tmp.txt

scp -r version_tmp.txt gc:$TARGET/version.txt

rm version_tmp.txt
