_base_ = '../default.py'

expname = 'dvgo_kitchen'
basedir = './logs/neural_rgbd_data'

data = dict(
    datadir='./data/neural_rgbd_data/kitchen',
    dataset_type='neuralrgbd',
    white_bkgd=False,
)
