#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "Usage: $0 version_name"
	exit 1
fi

VER=$1

TARGET="/home/matthew/playground/players/$VER"

ssh $HOST mkdir $TARGET

rsync -av --exclude '.hg' --exclude 'training' --exclude 'trainingResults' --include 'sts.epd' --exclude '*.epd' --exclude '*.epd.gz' --exclude '*.fen.gz' --exclude '*.log' --exclude '*.o' --exclude '*.a' --exclude '*.d' --exclude 'giraffe' . $TARGET/

echo $VER > version_tmp.txt

cp version_tmp.txt $TARGET/version.txt

rm version_tmp.txt
