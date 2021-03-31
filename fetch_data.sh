#!/bin/bash
mkdir -p data/UV_data
cd data/UV_data

# fetch DensePose UV data from facebookresearch/DensePose
wget https://dl.fbaipublicfiles.com/densepose/densepose_uv_data.tar.gz
tar xvf densepose_uv_data.tar.gz
rm densepose_uv_data.tar.gz
cd ../..

# fetch mesh_downsampling.npz from nkolot/GraphCMR
wget https://github.com/nkolot/GraphCMR/raw/master/data/mesh_downsampling.npz -O data/mesh_downsampling.npz
