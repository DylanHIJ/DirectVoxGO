_base_ = '../default.py'

expname = 'dvgo_staircase'
basedir = './logs/neural_rgbd_data'

data = dict(
    datadir='./data/neural_rgbd_data/staircase',
    dataset_type='neuralrgbd',
    white_bkgd=False,
)
