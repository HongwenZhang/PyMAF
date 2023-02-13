🚩 **[Update]** The face part of PyMAF-X has been updated. See face-only evaluation results on [the NoW benchmark](https://now.is.tue.mpg.de/nonmetricalevaluation.html).

<div align="center">

# PyMAF-X: Towards Well-aligned Full-body Model Regression from Monocular Images

[Hongwen Zhang](https://github.com/HongwenZhang) · [Yating Tian](https://github.com/tinatiansjz) · [Yuxiang Zhang](https://zhangyux15.github.io) · [Mengcheng Li](https://github.com/Dw1010) · [Liang An](https://anl13.github.io) · [Zhenan Sun](http://www.cbsr.ia.ac.cn/users/znsun) · [Yebin Liu](https://www.liuyebin.com)

### [Project Page](https://www.liuyebin.com/pymaf-x) | [Video](https://www.bilibili.com/video/BV1pN4y1T7dY) | [Paper](https://arxiv.org/abs/2207.06400)

</div>

<p align="center">
    <img src="https://hongwenzhang.github.io/pymaf-x/files/dance_demo.gif">
    <br>
    <sup>Frame by frame reconstruction. Video clipped from <a href="https://www.youtube.com/watch?v=Ltt4dkRkSG0" target="_blank"><i>here</i></a>.</sup>
    <br>
    <img src="https://hongwenzhang.github.io/pymaf-x/files/img_demo.png">
    <br>
    <sup>Reconstruction result on a COCO validation image.</sup>
    <br>
    <a href="https://www.liuyebin.com/pymaf-x" target="_blank"><i>Click Here</i></a> for More Results
</p>

## Installation

- Python 3.8

```
conda create --no-default-packages -n pymafx python=3.8
conda activate pymafx
```

### packages

- [PyTorch](https://www.pytorch.org) tested on version 1.9.0
```
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

- [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

- other packages listed in `requirements.txt`
```
pip install -r requirements.txt
```

### necessary files

> smpl_downsampling.npz & mano_downsampling.npz

- Run the following script to fetch necessary files.

```
bash fetch_data.sh
```
> SMPL & SMPL-X model files

- Collect [SMPL](https://smpl.is.tue.mpg.de)/[MANO](https://mano.is.tue.mpg.de)/[FLAME](https://flame.is.tue.mpg.de)/[SMPL-X](https://smpl-x.is.tue.mpg.de) model files. Rename model files and put them into the `./data/smpl` directory.

> Download the [partial_mesh](https://cloud.tsinghua.edu.cn/d/3bc20811a93b488b99a9) files and put it into the `./data/partial_mesh` directory.

> Download the [pre-trained model](https://cloud.tsinghua.edu.cn/d/3bc20811a93b488b99a9) and put it into the `./data/pretrained_model` directory.

After collecting the above necessary files, the directory structure of `./data` is expected as follows.  
```
./data
├── J_regressor_extra.npy
├── smpl_mean_params.npz
├── smpl_downsampling.npz
├── mano_downsampling.npz
├── flame_downsampling.npy
├── partial_mesh
│   └── ***_vids.npz
├── pretrained_model
│   └── PyMAF-X_model_checkpoint_v1.1.pt
└── smpl
    ├── FLAME2020
    │   ├── FLAME_NEUTRAL.pkl
    │   ├── flame_dynamic_embedding.npy
    │   └── flame_static_embedding.pkl
    ├── MANO_RIGHT.pkl
    ├── SMPLX_NEUTRAL_2020.npz
    ├── SMPL_NEUTRAL.pkl
    └── model_transfer
        ├── MANO_SMPLX_vertex_ids.pkl
        ├── SMPL-X__FLAME_vertex_ids.npy
        └── smplx_to_smpl.pkl
```

## Demo

You can first give it a try on Google Colab using the notebook we have prepared, which is no need to prepare the environment yourself: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13Iytx1Hb0ZryEwbJdpXBW9ggDxs2Y-tL?usp=sharing)

Run the demo code.

#### For image folder input:

```
python -m apps.demo_smplx --image_folder examples/coco_images --detection_threshold 0.3 --pretrained_model data/pretrained_model/PyMAF-X_model_checkpoint_v1.1.pt --misc TRAIN.BHF_MODE full_body MODEL.PyMAF.HAND_VIS_TH 0.1
```
#### For video input:
```
python -m apps.demo_smplx --vid_file examples/dancer_short.mp4 --pretrained_model data/pretrained_model/PyMAF-X_model_checkpoint_v1.1.pt --misc TRAIN.BHF_MODE full_body MODEL.PyMAF.HAND_VIS_TH 0.1
```

Results will be saved at `./output`. You can set different hyperparamters in the scripts, e.g., `--detection_threshold` for the person detection threshold and `MODEL.PyMAF.HAND_VIS_TH` for the hand visibility threshold.

## Citation
If this work is helpful in your research, please cite the following papers.
```
@article{pymafx2022,
  title={PyMAF-X: Towards Well-aligned Full-body Model Regression from Monocular Images},
  author={Zhang, Hongwen and Tian, Yating and Zhang, Yuxiang and Li, Mengcheng and An, Liang and Sun, Zhenan and Liu, Yebin},
  journal={arXiv preprint arXiv:2207.06400},
  year={2022}
}

@inproceedings{pymaf2021,
  title={PyMAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop},
  author={Zhang, Hongwen and Tian, Yating and Zhou, Xinchi and Ouyang, Wanli and Liu, Yebin and Wang, Limin and Sun, Zhenan},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2021}
}
```

## Acknowledgments

Part of the code is borrowed from the following projects, including [DaNet](https://github.com/HongwenZhang/DaNet-3DHumanReconstruction), [SPIN](https://github.com/nkolot/SPIN), [VIBE](https://github.com/mkocabas/VIBE), [SPEC](https://github.com/mkocabas/SPEC), [MeshGraphormer](https://github.com/microsoft/MeshGraphormer), [PIFu](https://github.com/shunsukesaito/PIFu), [DensePose](https://github.com/facebookresearch/DensePose), [HMR](https://github.com/akanazawa/hmr), [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch), [pose_resnet](https://github.com/Microsoft/human-pose-estimation.pytorch). Many thanks to their contributions.