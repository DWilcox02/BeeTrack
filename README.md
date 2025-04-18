# Computer Vision Bee Tracking - Final Year Thesis / Project
Computer Vision Bee Tracking - Imperial College London Final Year Thesis / Project. Detect and track honeybees performing their waggle dance for apiology researchers. Leverages TAPIR (Track Any Point library) for point cloud establishment, Kalman filtering and RANSAC for future position prediction. 

## Version Information
Running `python 3.11.10`. Updates made to fix Keras 3 compatibility.

## Initialization/Setup
After ensuring the same python version, run:
```
wget -P checkpoints/tapir https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.npy
wget -P checkpoints/bootstapir https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.npy
git submodule add https://github.com/deepmind/tapnet
cd tapnet
pip install .
cd ..
pip install dm-haiku jax mediapy numpy matplotlib tqdm tensorflow
```
or, run 
```
sh init.sh
```

TODO: Apply mkdir to the init video download

## Running
Move relevant videos into `/data/`. Run:
```
python src/tapir_bulk.py
```

### DoC GPU Cluster Guide
https://www.imperial.ac.uk/computing/people/csg/guides/hpcomputing/gpucluster/

## Next Steps
- Abstract use of GPUs. Currently running locally on laptop. Allow flexibility to run on a Google Cloud VM, DoC CSG GPU Cluster, etc..
    - Project must be cloned and run as a whole locally on VM


## Referenced Work
1. Bozek, K., Hebert, L., Portugal, Y. et al. Markerless tracking of an entire honey bee colony. Nat Commun 12, 1733 (2021). https://doi.org/10.1038/s41467-021-21769-1
    
    - Associated Github Repo: https://github.com/kasiabozek/bee_tracking/tree/master
    - Associated Annotation Repo: https://github.com/oist/DenseObjectAnnotation


2. Kongsilp, P., Taetragool, U. & Duangphakdee, O. Individual honey bee tracking in a beehive environment using deep learning and Kalman filter. Sci Rep 14, 1061 (2024). https://doi.org/10.1038/s41598-023-44718-y


<!-- This project includes code licensed under GPL-3.0 from [Original Repository](https://github.com/username/repository). -->




1. Prompt user to draw skeleton around bee
2. Determine bounding box based on skeleton
3. (Determine relationship between skeleton and points)
4. (Apply point cloud based on bounding box)
4. Run video for X time slice
5. Calculate estimated skeleton position via RANSAC and Kalman Filtering