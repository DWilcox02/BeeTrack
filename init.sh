#!/bin/bash

wget -P checkpoints/tapir https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.npy
wget -P checkpoints/bootstapir https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.npy

FILEID="1bMAUDTfqKsHQbCdk16_N2Ney7vLJm75E"
FILENAME="dance_15_secs_700x700_50fps.mp4"

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O $FILENAME && rm -rf /tmp/cookies.txt# https://drive.google.com/file/d/1bMAUDTfqKsHQbCdk16_N2Ney7vLJm75E/view?usp=drive_link