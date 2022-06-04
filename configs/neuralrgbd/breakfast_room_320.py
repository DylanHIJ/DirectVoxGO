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