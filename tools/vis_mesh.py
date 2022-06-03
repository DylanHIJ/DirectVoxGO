import os
import argparse
import numpy as np
import open3d as o3d

H, W = 480, 640

def load_pose(pose_file, pose_idx):
    with open(pose_file, "r") as f:
        lines = f.readlines()
        pose = [[float(x) for x in line.split()] for line in lines[pose_idx*4:(pose_idx+1)*4]]
    return np.array(pose)


def load_focal_length(filepath):
    file = open(filepath, "r")
    return float(file.readline())

def main(args):
    mesh = o3d.io.read_triangle_mesh(args.input)
    mesh.compute_vertex_normals()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=W, height=H)
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    vis.get_render_option().mesh_show_back_face = True
    vis.add_geometry(mesh)

    pose = load_pose(os.path.join(args.basedir, "poses.txt"), args.pose_idx)
    pose = -np.linalg.inv(pose)
    focal = load_focal_length(os.path.join(args.basedir, 'focal.txt'))

    view_ctl = vis.get_view_control()
    cam = view_ctl.convert_to_pinhole_camera_parameters()
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(W, H, focal, focal, 0.5 * W, 0.5 * H)
    cam.extrinsic = pose
    cam.intrinsic = intrinsic
    view_ctl.convert_from_pinhole_camera_parameters(cam, True)

    vis.run()
    if args.output:
        vis.capture_screen_image(args.output)
    vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path of input .ply file")
    parser.add_argument("--output", default="", help="path of output rendered image")
    parser.add_argument(
        "--basedir",
        required=True,
        help="base directory for data, which should contain poses.txt and focal.txt"
    )
    parser.add_argument("--pose_idx", default=0, type=int, help="index of camera pose to be used")
    args = parser.parse_args()

    main(args)