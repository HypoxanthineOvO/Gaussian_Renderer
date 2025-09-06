import numpy as np
from plyfile import PlyData

import torch
import math

def invert_cov_2D(cov_2D: torch.Tensor):
    """
    Using the formula for inverse of a 2D matrix to invert the cov_2D matrix

    Args:
        cov_2D          :   A torch.Tensor of shape (N, 2, 2)

    Returns:
        cov_2D_inverse  :   A torch.Tensor of shape (N, 2, 2)
    """
    determinants = cov_2D[:, 0, 0] * cov_2D[:, 1, 1] - cov_2D[:, 1, 0] * cov_2D[:, 0, 1]
    determinants = determinants[:, None, None]  # (N, 1, 1)

    cov_2D_inverse = torch.zeros_like(cov_2D)  # (N, 2, 2)
    cov_2D_inverse[:, 0, 0] = cov_2D[:, 1, 1]
    cov_2D_inverse[:, 1, 1] = cov_2D[:, 0, 0]
    cov_2D_inverse[:, 0, 1] = -1.0 * cov_2D[:, 0, 1]
    cov_2D_inverse[:, 1, 0] = -1.0 * cov_2D[:, 1, 0]

    cov_2D_inverse = (1.0 / determinants) * cov_2D_inverse

    return cov_2D_inverse

def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def compute_jacobian(
    means_3D: torch.Tensor, 
    focal_length, 
    view_transform,  
    img_size):

    #fx, fy = camera.focal_length.flatten()
    fx, fy = focal_length
    W, H = img_size

    half_tan_fov_x = 0.5 * W / fx
    half_tan_fov_y = 0.5 * H / fy

    #view_transform = camera.get_world_to_view_transform()
    means_view_space = view_transform.transform_points(means_3D)

    tx = means_view_space[:, 0]
    ty = means_view_space[:, 1]
    tz = means_view_space[:, 2]
    tz2 = tz*tz

    lim_x = 1.3 * half_tan_fov_x
    lim_y = 1.3 * half_tan_fov_y

    tx = torch.clamp(tx/tz, -lim_x, lim_x) * tz
    ty = torch.clamp(ty/tz, -lim_y, lim_y) * tz

    J = torch.zeros((len(tx), 2, 3), device = means_3D.device)  # (N, 2, 3)

    J[:, 0, 0] = fx / tz
    J[:, 1, 1] = fy / tz
    J[:, 0, 2] = -(fx * tx) / tz2
    J[:, 1, 2] = -(fy * ty) / tz2

    return J  # (N, 2, 3)

def look_at_view_transform(dist: float, azim: float, elev: float, up: tuple = (0, -1, 0)):
    """
    计算相机看向原点的旋转矩阵和平移向量
    参数:
        dist: 相机到原点的距离 (标量)
        azim: 方位角 (度), 在XZ平面上从正X轴开始的角度
        elev: 仰角 (度), 从XZ平面向上/向下的角度
        up: 上方向向量 (元组或张量)
    返回:
        R: 3x3 旋转矩阵 (torch.Tensor)
        T: 3D 平移向量 (torch.Tensor)
    """
    # 将角度转换为弧度
    azim_rad = math.radians(azim)
    elev_rad = math.radians(elev)
    
    # 计算相机位置 (球坐标转笛卡尔坐标)
    # 注意：PyTorch3D 使用 Y-up 坐标系
    x = dist * math.cos(elev_rad) * math.cos(azim_rad)
    y = dist * math.sin(elev_rad)
    z = dist * math.cos(elev_rad) * math.sin(azim_rad)
    camera_position = torch.tensor([x, y, z])
    
    # 标准化上方向向量
    up_vector = torch.tensor(up, dtype=torch.float32)
    up_vector = up_vector / torch.norm(up_vector)
    
    # 计算相机坐标系各轴方向
    # z轴：从相机指向原点 (0,0,0)
    z_axis = -camera_position  # 指向原点
    z_axis = z_axis / torch.norm(z_axis)
    
    # x轴：上方向与z轴的叉积 (右手坐标系)
    x_axis = torch.cross(up_vector, z_axis)
    # 处理叉积接近零的情况 (当z轴与上方向平行时)
    if torch.norm(x_axis) < 1e-6:
        # 使用备用上方向 (0, 0, 1) 重新计算
        backup_up = torch.tensor([0, 0, 1], dtype=torch.float32)
        x_axis = torch.cross(backup_up, z_axis)
    x_axis = x_axis / torch.norm(x_axis)
    
    # y轴：z轴与x轴的叉积 (确保三个轴正交)
    y_axis = torch.cross(z_axis, x_axis)
    y_axis = y_axis / torch.norm(y_axis)
    
    # 构建旋转矩阵 (相机坐标系到世界坐标系的变换)
    R = torch.stack([x_axis, y_axis, z_axis], dim=0)
    
    # 计算平移向量 (相机在世界坐标系中的位置取负)
    T = -R @ camera_position
    
    return R, T


def load_gaussians_from_ply(path):
    # Modified from https://github.com/thomasantony/splat
    max_sh_degree = 3
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = np.transpose(features_extra, [0, 2, 1])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    xyz = xyz.astype(np.float32)
    rots = rots.astype(np.float32)
    scales = scales.astype(np.float32)
    opacities = opacities.astype(np.float32)
    shs = np.concatenate([
        features_dc.reshape(-1, 3),
        features_extra.reshape(len(features_dc), -1)
    ], axis=-1).astype(np.float32)
    shs = shs.astype(np.float32)

    dc_vals = shs[:, :3]
    dc_colours = np.maximum(dc_vals * 0.28209479177387814 + 0.5, np.zeros_like(dc_vals))

    output = {
        "xyz": xyz, "rot": rots, "scale": scales,
        "sh": shs, "opacity": opacities, "dc_colours": dc_colours
    }
    return output