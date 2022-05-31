import os
import mcubes
import trimesh
import torch
import numpy as np

def get_scene_bounds(xyz_min, xyz_max, voxel_size, interp=False):
    x_min, y_min, z_min = xyz_min.detach().cpu().numpy()
    x_max, y_max, z_max = xyz_max.detach().cpu().numpy()

    if interp:
        x_min = x_min - 0.5 * voxel_size
        y_min = y_min - 0.5 * voxel_size
        z_min = z_min - 0.5 * voxel_size

        x_max = x_max + 0.5 * voxel_size
        y_max = y_max + 0.5 * voxel_size
        z_max = z_max + 0.5 * voxel_size

    Nx = round((x_max - x_min) / voxel_size + 0.0005)
    Ny = round((y_max - y_min) / voxel_size + 0.0005)
    Nz = round((z_max - z_min) / voxel_size + 0.0005)

    tx = np.linspace(x_min, x_max, Nx + 1)
    ty = np.linspace(y_min, y_max, Ny + 1)
    tz = np.linspace(z_min, z_max, Nz + 1)

    return tx, ty, tz

def extract_iso_level(density):
    # Density boundaries
    min_a, max_a, std_a = density.min(), density.max(), density.std()

    # Adaptive iso level
    iso_value = min(max(32, min_a + std_a), max_a - std_a)
    print(f"Min density {min_a}, Max density: {max_a}, Mean density {density.mean()}")
    print(f"Querying based on iso level: {iso_value}")

    return iso_value

def extract_mesh(model, ndc, render_kwargs, voxel_size=0.01, savedir=None):
    print("extract_mesh started")
    # voxel_size *= 0.4
    tx, ty, tz = get_scene_bounds(model.xyz_min, model.xyz_max, voxel_size, True)

    query_pts = np.stack(np.meshgrid(tx, ty, tz, indexing='ij'), -1).astype(np.float32)
    print(f"shape of query_pts: {query_pts.shape}")
    shape = query_pts.shape
    flat_query_pts = query_pts.reshape([-1, 3])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flat_query_pts = torch.from_numpy(flat_query_pts).to(device)

    ret = model.grid_sampler(flat_query_pts, model.density)
    density = np.reshape(ret.detach().cpu().numpy(), shape[:-1])

    tmp = extract_iso_level(density)

    for iso_level in range(16, 0, -1):
        vertices, triangles = mcubes.marching_cubes(density, iso_level)

        # normalize vertex positions
        vertices[:, :3] /= np.array([[tx.shape[0] - 1, ty.shape[0] - 1, tz.shape[0] - 1]])

        # Rescale and translate
        scale = np.array([tx[-1] - tx[0], ty[-1] - ty[0], tz[-1] - tz[0]])
        offset = np.array([tx[0], ty[0], tz[0]])
        vertices[:, :3] = scale[np.newaxis, :] * vertices[:, :3] + offset

        mesh = trimesh.Trimesh(vertices, triangles, process=False)

        mesh_savepath = os.path.join(savedir, f"mesh_{iso_level}.ply")
        mesh.export(mesh_savepath)

    print("extract_mesh ended")
