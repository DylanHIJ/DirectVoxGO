_base_ = '../default.py'

expname = 'dvgo_breakfast_room'
basedir = './logs/neural_rgbd_data'

data = dict(
    datadir='./data/neural_rgbd_data/breakfast_room',
    dataset_type='neuralrgbd',
    white_bkgd=False,
)

