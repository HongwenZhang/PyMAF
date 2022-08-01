#!/bin/bash

# Script that fetches necessary data

# fetch model constants etc. from nkolot/GraphCMR
wget http://visiondata.cis.upenn.edu/spin/data.tar.gz && tar -xvf data.tar.gz && rm data.tar.gz

# fetch smpl_downsampling.npz from nkolot/GraphCMR
wget https://github.com/nkolot/GraphCMR/raw/master/data/mesh_downsampling.npz -O data/smpl_downsampling.npz

# fetch mano_downsampling.npz from microsoft/MeshGraphormer
wget https://github.com/microsoft/MeshGraphormer/raw/main/src/modeling/data/mano_downsampling.npz -O data/mano_downsampling.npz

