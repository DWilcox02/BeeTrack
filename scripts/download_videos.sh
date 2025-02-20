#!/bin/bash

mkdir -p data/Dance_1_min

FILEID="1bMAUDTfqKsHQbCdk16_N2Ney7vLJm75E"
FILENAME="data/Dance_1_min/dance_15_secs_700x700_50fps.mp4"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O $FILENAME && rm -rf /tmp/cookies.txt


FILEID="1q6UwpIu29Y1zUQxXkkwIsV4vP2h4QQOi"
FILENAME="data/Dance_1_min/dance_7_point_5_secs_700x700_30fps.mp4"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O $FILENAME && rm -rf /tmp/cookies.txt