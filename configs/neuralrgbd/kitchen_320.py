_base_ = '../default.py'

expname = 'dvgo_kitchen_320'
basedir = './logs/neural_rgbd_data'

data = dict(
    datadir='./data/neural_rgbd_data/kitchen',
    dataset_type='neuralrgbd',
    white_bkgd=False,
)

fine_model_and_render = dict(
    num_voxels=320**3,
)
