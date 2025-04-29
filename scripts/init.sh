#!/bin/bash

wget -P checkpoints/tapir https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.npy
wget -P checkpoints/bootstapir https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.npy

git submodule add https://github.com/DWilcox02/tapnet_beetrack
cd tapnet
pip install .
cd ..

cd src/frontend
npm install
cd ../..

pip install dm-haiku jax mediapy numpy matplotlib tqdm tensorflow