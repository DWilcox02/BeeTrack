#!/bin/bash

FILEID="1bMAUDTfqKsHQbCdk16_N2Ney7vLJm75E"
FILENAME="data/Dance_1_min/dance_15_secs_700x700_50fps.mp4"

mkdir -p data/Dance_1_min
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O $FILENAME && rm -rf /tmp/cookies.txt