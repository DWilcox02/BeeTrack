# Computer Vision Bee Tracking - Final Year Thesis / Project :bee:

Computer Vision Bee Tracking - Imperial College London Final Year Thesis / Project. Detect and track honeybees performing their waggle dance for apiology researchers. Leverages TAPIR (Track Any Point library) for point cloud establishment, Kalman filtering and RANSAC for future position prediction. 

<p align="center">
    <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" alt="python"> <img src="https://img.shields.io/badge/JavaScript-323330?style=for-the-badge&logo=javascript&logoColor=F7DF1E" alt="JavaScript"> <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white" alt="TensorFlow"> <img src="https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly">
</p>
<p align="center">
    <img src="https://img.shields.io/badge/GPL--3.0-red?style=for-the-badge" alt="GPL3">
</p>

## Version Information
Running `python 3.11.10`. Updates made to fix Keras 3 compatibility.

Note: This was all run on macOS (Sequoia 15.3).

## Initialization/Setup
After ensuring the same python version (e.g. after setting up an appropriate python virtual environment), either run `scripts/init.sh` for initialization, or run the 
following commands in each section:


1. Install TAPIR checkpoints (in root folder)
    ```
    wget -P checkpoints/tapir https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.npy
    wget -P checkpoints/bootstapir https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.npy
    ```

3. Install my forked Tapnet and its dependencies
    ```
    git submodule add https://github.com/DWilcox02/tapnet_beetrack
    cd tapnet
    pip install .
    cd ..
    ```

4. Install node modules
    ```
    cd src/frontend
    npm install
    cd ../..
    ```

5. Install backend dependencies
    ```
    pip install dm-haiku jax mediapy numpy matplotlib tqdm tensorflow
    ```
    (Note: this may be incomplete and you'll have to install further python packages)

###   (Optional) Download sample videos 
If you don't have any videos for testing, download honeybee videos with
```
`scripts/download_videos.sh`
```

## Running
### Setup backend Flask server
From root, run:
```
python -m src.backend.app
```
to start the backend flask server, at `http://127.0.0.1:5001`

### Setup frontend JS/HTML server
From `src/frontend`, run:
```
npm run dev
```
to start the frontend server, at `http://localhost:3000`


## Misc.

### Progress Outline
1. TAPIR wrapper for running locally + video compression
2. Flask server for user interface
3. Plotly graph for point selection
4. Point interpolation across bee
5. Basic re-adjustment process, using "segment" midpoint and trajectory to recalculate points after each segment
6. Server split into JS/HTML (frontend) + Python (backend) with socket for bidirectional communication
7. User validation, allowing point updates after each "segment"

### Next Steps
- Establish "uncertainty predicate" to measure how uncertain the point cloud estimate is of the bee in any given frame.
- Replace primitive "midpoint + trajectory"-based recalculation with Kalman Filtering and RANSAC
    - Necessary to outline where and how precisely these techniques will be used
- Normalize video FPS to 15. Any higher results in unnecessarly heavy computation
    - Potential for dynamic framerate in cases of uncertainty?
- Ensure linux compatibility

### DoC GPU Cluster Guide
https://www.imperial.ac.uk/computing/people/csg/guides/hpcomputing/gpucluster/


## Referenced Work
1. Bozek, K., Hebert, L., Portugal, Y. et al. Markerless tracking of an entire honey bee colony. Nat Commun 12, 1733 (2021). https://doi.org/10.1038/s41467-021-21769-1
    
    - Associated Github Repo: https://github.com/kasiabozek/bee_tracking/tree/master
    - Associated Annotation Repo: https://github.com/oist/DenseObjectAnnotation


2. Kongsilp, P., Taetragool, U. & Duangphakdee, O. Individual honey bee tracking in a beehive environment using deep learning and Kalman filter. Sci Rep 14, 1061 (2024). https://doi.org/10.1038/s41598-023-44718-y


<!-- This project includes code licensed under GPL-3.0 from [Original Repository](https://github.com/username/repository). -->



<!-- 
1. Prompt user to draw skeleton around bee
2. Determine bounding box based on skeleton
3. (Determine relationship between skeleton and points)
4. (Apply point cloud based on bounding box)
4. Run video for X time slice
5. Calculate estimated skeleton position via RANSAC and Kalman Filtering -->