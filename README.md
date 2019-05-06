# Semantic Instance Geometry Network for Unsupervised Percepetion

> Project page: [https://mengyuest.github.io/SIGNet/](https://mengyuest.github.io/SIGNet/)

This is the implementation of our paper:


Y. Meng, Y. Lu, A. Raj, S. Sunarjo, R. Guo, T. Javidi, G. Bansal, D. Bharadia. **"SIGNet: Semantic Instance Aided Unsupervised 3D Geometry Perception"**,  (CVPR), 2019. \[[arXiv pdf](https://arxiv.org/pdf/1812.05642.pdf)\] 


The code is build upon [GeoNet](https://github.com/yzcjtr/GeoNet)


## Prerequisite

1. Ubuntu 16.04, python3, tensorflow-gpu 1.10.1 (test on GTX 1080Ti and RTX 2080Ti with CUDA 9.0)
2. Better to use virtual environment. For the rest of dependencies, please run `pip3 install -r requirements.txt`)
3. Download ground truth depth and our models from [https://drive.google.com/open?id=19BFkrfODd3N5IKQJJgqp-pXjbHeYrFf1](https://drive.google.com/open?id=19BFkrfODd3N5IKQJJgqp-pXjbHeYrFf1) (put the `models `folder directly under the project directory)
4. Download KITTI evaluation dataset from [https://drive.google.com/open?id=1kYNKqIhArAD03WNr4_FZCYRRWo0WT31P](https://drive.google.com/open?id=1kYNKqIhArAD03WNr4_FZCYRRWo0WT31P) (move it two levels upon the project directory, i.e. `mv -f data ../../data`)


## Inference for Depth

1. Run `bash run_all_tests.sh` then wait for 2~4 minutes. Results are related to Table 1 ~ Table 4 in our paper.


## Training on [KITTI](http://www.cvlibs.net/datasets/kitti/index.php)

1. Follow the **Data preparation** instructions from [GeoNet](https://github.com/yzcjtr/GeoNet).
2. Prepare for semantic lables (semantic-level: [DeeplabV3+](https://github.com/tensorflow/models/tree/master/research/deeplab), instance-level: [Mask-RCNN](https://github.com/facebookresearch/Detectron))
3. Quick training: run `bash run_depth_train.sh config/foobar.cfg` where `foobar.cfg` is the configuration filename you need to specify.
4. Logs will be saved in `${CHECKPOINT_DIR}/logs/` defined in `foobar.cfg` file
