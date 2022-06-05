# Super-fast Convergence for 3D Scene Reconstruction

This work reconstructs a scene representation from a set of calibrated RGB-D images capturing the scene, which is based on the following works:

1. Azinović, Dejan, et al. "Neural RGB-D surface reconstruction." arXiv preprint arXiv:2104.04532 (2021).

   ([Project Page](https://dazinovic.github.io/neural-rgbd-surface-reconstruction/))

2. Sun, Cheng, Min Sun, and Hwann-Tzong Chen. "Direct Voxel Grid Optimization: Super-fast Convergence for Radiance Fields Reconstruction." arXiv preprint arXiv:2111.11215 (2021).

   ([Project Page](https://sunset1995.github.io/dvgo/))

![](https://i.imgur.com/mwkg2P1.png)

(Image Credit to [Direct Voxel Grid Optimization: Super-fast Convergence for Radiance Fields Reconstruction.](https://arxiv.org/pdf/2111.11215.pdf))

## Installation

```bash
pip install -r requirements.txt
```
[Pytorch](https://pytorch.org/) and [torch_scatter](https://github.com/rusty1s/pytorch_scatter) installation is machine dependent, please install the correct version for your machine.

## Dataset

We use the data provided by [Neural RGB-D surface reconstruction](https://dazinovic.github.io/neural-rgbd-surface-reconstruction/), which contains RGB-D images, ground-truth poses and camera intrinsics for 9 different scenes. The full dataset can be downloaded from [neural_rgbd_data.zip](http://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip).


## How to Run

### Train and Extract
To train on a scene and extract mesh from the resulting model, run: 

```bash
$ python run.py --config [config file] --render_test
# e.g python run.py --config configs/neuralrgbd/breakfast_room.py --render_test
```
Use `--i_print` and `--i_weights` to change the log interval.

### Your own config files
Check the comments in [`configs/default.py`](./configs/default.py) for the configuable settings.
We use [`mmcv`'s config system](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html).
To create a new config, please inherit `configs/default.py` first and then update the fields you want.
Below is an example from `configs/neuralrgbd/breakfast_room_320.py`:

```python
_base_ = '../default.py'

expname = 'dvgo_breakfast_room_320'
basedir = './logs/neural_rgbd_data'

data = dict(
    datadir='./data/neural_rgbd_data/breakfast_room',
    dataset_type='neuralrgbd',
    white_bkgd=False,
)

fine_model_and_render = dict(
    num_voxels=320**3,
)
```

## Acknowledgement
The code base is forked from [DirectVoxelGO](https://github.com/sunset1995/DirectVoxGO) and much modification is based on [Neural RGB-D Surface Reconstruction](https://github.com/dazinovic/neural-rgbd-surface-reconstruction).
