# Computer Vision Bee Tracking - Final Year Project
Computer Vision Bee Tracking - Imperial College London Final Year Project. Detect and track honeybees performing their waggle dance for apiology researchers. 

## Version Information
Running `python 3.11.10`. Updates made to fix Keras 3 compatibility.

## Initialization/Setup
After ensuring the same python version, run:
```
wget -P checkpoints/bootstapir https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.npy
wget -P checkpoints/tapir https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.npy
git submodule add https://github.com/deepmind/tapnet
cd tapnet
pip install .
cd ..
pip install -r requirements.txt
```
or, run 
```
sh init.sh
```

## Running
Move relevant videos into `/data/`. Run:
```
python src/tapir_bulk.py
```

## Referenced Work
1. Bozek, K., Hebert, L., Portugal, Y. et al. Markerless tracking of an entire honey bee colony. Nat Commun 12, 1733 (2021). https://doi.org/10.1038/s41467-021-21769-1
    
    - Associated Github Repo: https://github.com/kasiabozek/bee_tracking/tree/master
    - Associated Annotation Repo: https://github.com/oist/DenseObjectAnnotation


2. Kongsilp, P., Taetragool, U. & Duangphakdee, O. Individual honey bee tracking in a beehive environment using deep learning and Kalman filter. Sci Rep 14, 1061 (2024). https://doi.org/10.1038/s41598-023-44718-y


This project includes code licensed under GPL-3.0 from [Original Repository](https://github.com/username/repository).
