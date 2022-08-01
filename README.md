🚩 **[Update]** Demo code for PyMAF-X is now released! Please check out the `smplx` branch [here](https://github.com/HongwenZhang/PyMAF/tree/smplx) `git checkout smplx`

🚩 **[Update]** PyMAF is now extended as [PyMAF-X](https://www.liuyebin.com/pymaf-x) for full-body model recovery! Please check out the new project page [here](https://www.liuyebin.com/pymaf-x)

# PyMAF & PyMAF-X
This repository contains the code for the following papers:

**PyMAF-X: Towards Well-aligned Full-body Model Regression from Monocular Images**  
Hongwen Zhang, Yating Tian, Yuxiang Zhang, Mengcheng Li, Liang An, Zhenan Sun, Yebin Liu 

arXiv, 2022

[[Project Page]](https://www.liuyebin.com/pymaf-x) [[Paper]](https://arxiv.org/pdf/2207.06400.pdf) [[Code: smplx branch]](https://github.com/HongwenZhang/PyMAF/tree/smplx)

**PyMAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop**  
Hongwen Zhang*, Yating Tian*, Xinchi Zhou, Wanli Ouyang, Yebin Liu, Limin Wang, Zhenan Sun 

\* Equal contribution

ICCV, 2021 (Oral Paper)

[[Project Page]](https://hongwenzhang.github.io/pymaf) [[Paper]](https://arxiv.org/pdf/2103.16507.pdf) [Code: smpl branch]

# Instruction for PyMAF

<!-- [![PyMAF](https://hongwenzhang.github.io/images/pymaf.jpg "PyMAF")](https://hongwenzhang.github.io/pymaf) -->

Preview of demo results:

<p align="left">
    <img src="https://hongwenzhang.github.io/pymaf/files/flashmob.gif">
    <br>
    <sup>Frame by frame reconstruction. Video clipped from <a href="https://www.youtube.com/watch?v=2DiQUX11YaY" target="_blank"><i>here</i></a>.</sup>
</p>
<p align="left">
    <img src="https://hongwenzhang.github.io/pymaf/files/dancer_result.gif">
    <br>
    <sup>Frame by frame reconstruction. Video clipped from <a href="https://youtu.be/xg_nXRHoqkE?t=65" target="_blank"><i>here</i></a>.</sup>
</p>

More results: [Click Here](https://hongwenzhang.github.io/pymaf)

## Requirements

- Python 3.6.10

### packages

- [PyTorch](https://www.pytorch.org) tested on version 1.1.0

- [torchvision](https://www.pytorch.org) tested on version 0.3.0

- [Neural Renderer](https://github.com/daniilidis-group/neural_renderer) (render densepose labels for training)

- [opendr](https://gitlab.eecs.umich.edu/ngv-python-modules/opendr#) (visualization in training)

- [pyrender](https://github.com/mmatl/pyrender) (optional for demo)

- other packages listed in `requirements.txt`

### necessary files

> mesh_downsampling.npz & DensePose UV data

- Run the following script to fetch mesh_downsampling.npz & DensePose UV data from other repositories.

```
bash fetch_data.sh
```
> SMPL model files

- Collect SMPL model files from [https://smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) and [UP](https://github.com/classner/up/blob/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl). Rename model files and put them into the `./data/smpl` directory.

> Fetch preprocessed data from [SPIN](https://github.com/nkolot/SPIN#fetch-data).

> Fetch final_fits data from [SPIN](https://github.com/nkolot/SPIN#final-fits).

> Download the [pre-trained model](https://drive.google.com/drive/folders/1R4_Vi4TpCQ26-6_b2PhjTBg-nBxZKjz6?usp=sharing) and put it into the `./data/pretrained_model` directory.

After collecting the above necessary files, the directory structure of `./data` is expected as follows.  
```
./data
├── dataset_extras
│   └── .npz files
├── J_regressor_extra.npy
├── J_regressor_h36m.npy
├── mesh_downsampling.npz
├── pretrained_model
│   └── PyMAF_model_checkpoint.pt
├── smpl
│   ├── SMPL_FEMALE.pkl
│   ├── SMPL_MALE.pkl
│   └── SMPL_NEUTRAL.pkl
├── smpl_mean_params.npz
├── final_fits
│   └── .npy files
└── UV_data
    ├── UV_Processed.mat
    └── UV_symmetry_transforms.mat
```

## Demo
[UPDATE] You can first give it a try on Google Colab using the notebook we have prepared, which is no need to prepare the environment yourself: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11RXLsH9BdoSCwY6G-IX7KgqDxVoImu6K?usp=sharing)

Run the demo code.

#### For image input:

```
python3 demo.py --checkpoint=data/pretrained_model/PyMAF_model_checkpoint.pt --img_file examples/COCO_val2014_000000019667.jpg
```
#### For video input:
```
# video with single person
python3 demo.py --checkpoint=data/pretrained_model/PyMAF_model_checkpoint.pt --vid_file examples/dancer.mp4
# video with multiple persons
python3 demo.py --checkpoint=data/pretrained_model/PyMAF_model_checkpoint.pt --vid_file examples/flashmob.mp4
```


## Evaluation

### COCO Keypoint Localization

1. Download the preprocessed data [coco_2014_val.npz](https://drive.google.com/drive/folders/1R4_Vi4TpCQ26-6_b2PhjTBg-nBxZKjz6?usp=sharing). Put it into the `./data/dataset_extras` directory. 

2. Run the COCO evaluation code.
```
python3 eval_coco.py --checkpoint=data/pretrained_model/PyMAF_model_checkpoint.pt
```

Results in [Average Precision (AP)](https://cocodataset.org/#keypoints-eval):

| Method         | AP &uarr; | AP<sub>50</sub> &uarr; | AP<sub>75</sub> &uarr; | AP<sub>M</sub> &uarr; |   AP<sub>L</sub> &uarr; |
|----------------|:------------:|:------------:|:------------:|:------------:|:------------:|
| [HMR](https://github.com/akanazawa/hmr)     | 18.9 |   47.5 | 11.7 |  21.5 |  17.0 |  
| [SPIN](https://github.com/nkolot/SPIN)     | 17.3 |   39.1  |  13.5 |   19.0  | 16.6  |
| Baseline | 16.8  |  38.2 | 12.8 |  18.5   | 16.0 | 
| PyMAF    | **24.6** |    **48.9**  | **22.7** |  **26.0** | **24.2** | 

### Human3.6M / 3DPW

Run the evaluation code. Using `--dataset` to specify the evaluation dataset.
```
# Example usage:

# Human3.6M Protocol 2
python3 eval.py --checkpoint=data/pretrained_model/PyMAF_model_checkpoint.pt --dataset=h36m-p2 --log_freq=20

# 3DPW
python3 eval.py --checkpoint=data/pretrained_model/PyMAF_model_checkpoint.pt --dataset=3dpw --log_freq=20
```

Results in Mean Per Joint Position Error (MPJPE):

| Method         | 3DPW &darr; | H36M &darr; |
|----------------|:------------:|:------------:|
| [SPIN](https://github.com/nkolot/SPIN)       | 96.9 |    62.5 |
| [VIBE](https://github.com/mkocabas/VIBE)      | 93.5 |    65.9  |
| Baseline | 98.5 |  64.8 |
| PyMAF    | **92.8** |    **57.7**  |

## Training

To perform training, we need to collect preprocessed files of training datasets at first.

The preprocessed labels have the same format as SPIN and can be retrieved from [here](https://github.com/nkolot/SPIN#fetch-data). Please refer to [SPIN](https://github.com/nkolot/SPIN) for more details about data preprocessing.

PyMAF is trained on Human3.6M at the first stage and then trained on the mixture of both 2D and 3D datasets at the second stage. Example usage:
```
# training on Human3.6M
CUDA_VISIBLE_DEVICES=0 python3 train.py --regressor pymaf_net --single_dataset --misc TRAIN.BATCH_SIZE 64
# training on mixed datasets
CUDA_VISIBLE_DEVICES=0 python3 train.py --regressor pymaf_net --pretrained_checkpoint path/to/checkpoint_file.pt --misc TRAIN.BATCH_SIZE 64
```
Running the above commands will use Human3.6M or mixed datasets for training, respectively. We can monitor the training process by setting up a TensorBoard at the directory `./logs`.

## Citation
If this work is helpful in your research, please cite the following paper.
```
@inproceedings{pymaf2021,
  title={PyMAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop},
  author={Zhang, Hongwen and Tian, Yating and Zhou, Xinchi and Ouyang, Wanli and Liu, Yebin and Wang, Limin and Sun, Zhenan},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2021}
}

@article{pymafx2022,
  title={PyMAF-X: Towards Well-aligned Full-body Model Regression from Monocular Images},
  author={Zhang, Hongwen and Tian, Yating and Zhang, Yuxiang and Li, Mengcheng and An, Liang and Sun, Zhenan and Liu, Yebin},
  journal={arXiv preprint arXiv:2207.06400},
  year={2022}
}
```

## Acknowledgments

The code is developed upon the following projects. Many thanks to their contributions.

- [SPIN](https://github.com/nkolot/SPIN)

- [DaNet](https://github.com/HongwenZhang/DaNet-3DHumanReconstruction)

- [VIBE](https://github.com/mkocabas/VIBE)

- [PIFu](https://github.com/shunsukesaito/PIFu)

- [DensePose](https://github.com/facebookresearch/DensePose)

- [HMR](https://github.com/akanazawa/hmr)

- [pose_resnet](https://github.com/Microsoft/human-pose-estimation.pytorch)
