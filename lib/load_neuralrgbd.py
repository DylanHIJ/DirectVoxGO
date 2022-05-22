import os
import re
import glob
import numpy as np
import imageio
import cv2

def load_poses(posefile):
    file = open(posefile, "r")
    lines = file.readlines()
    file.close()
    poses = []
    valid = []
    lines_per_matrix = 4
    for i in range(0, len(lines), lines_per_matrix):
        if 'nan' in lines[i]:
            valid.append(False)
            poses.append(np.eye(4, 4, dtype=np.float32).tolist())
        else:
            valid.append(True)
            pose_floats = [[float(x) for x in line.split()] for line in lines[i:i+lines_per_matrix]]
            poses.append(pose_floats)

    return poses, valid

def load_focal_length(filepath):
    file = open(filepath, "r")
    return float(file.readline())

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [int(x) if x.isdigit() else x for x in re.split('([0-9]+)', s)]

def resize_images(images, H, W, interpolation=cv2.INTER_LINEAR):
    resized = np.zeros((images.shape[0], H, W, images.shape[3]), dtype=images.dtype)
    for i, img in enumerate(images):
        r = cv2.resize(img, (W, H), interpolation=interpolation)
        if images.shape[3] == 1:
            r = r[..., np.newaxis]
        resized[i] = r
    return resized

def load_neuralrgbd_data(basedir):
    rgb_paths = [os.path.join(basedir, "images", f)
            for f in sorted(os.listdir(os.path.join(basedir, 'images')), key=alphanum_key) if f.endswith('png')]
    depth_paths = [os.path.join(basedir, "depth_filtered", f)
            for f in sorted(os.listdir(os.path.join(basedir, 'depth_filtered')), key=alphanum_key) if f.endswith('png')]
    all_poses, valid_poses = load_poses(os.path.join(basedir, 'poses.txt'))


    all_imgs, all_depths = [], []
    i_split = [[], []]
    for i, (rgb_path, depth_path) in enumerate(zip(rgb_paths, depth_paths)):
        i_set = 1 if i % 8 == 0 else 0
        all_imgs.append((imageio.imread(rgb_path) / 255.).astype(np.float32))
        all_depths.append((imageio.imread(depth_path) / 1000.).astype(np.float32))
        i_split[i_set].append(i)

    imgs = np.stack(all_imgs, 0)
    depths = np.stack(all_depths, 0)
    depths = depths[..., np.newaxis]

    poses = np.stack(all_poses, 0).astype(np.float32)
    i_split.append(i_split[-1])

    H, W = imgs[0].shape[:2]
    focal = load_focal_length(os.path.join(basedir, 'focal.txt'))

    render_poses = poses[i_split[-1]]

    return imgs, depths, poses, render_poses, [H, W, focal], i_split