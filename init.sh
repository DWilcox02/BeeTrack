wget -P checkpoints/bootstapir https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.npy
wget -P checkpoints/tapir https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.npy
git submodule add https://github.com/deepmind/tapnet
cd tapnet
pip install .
cd ..
pip install dm-haiku jax mediapy numpy matplotlib