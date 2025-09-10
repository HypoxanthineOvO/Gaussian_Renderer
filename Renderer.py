import numpy as np
import math
import torch

#from pytorch3d.renderer.cameras import PerspectiveCameras as PerspectiveCamera
from Camera import PerspectiveCamera
from Gaussian import GaussianModel
from utils import compute_jacobian, quat_to_mat, invert_cov_2D

from tqdm import trange, tqdm

def splat(
    camera: PerspectiveCamera,
    means_3d: torch.Tensor, z_vals: torch.Tensor, quats: torch.Tensor,
    scales: torch.Tensor, opacities: torch.Tensor, colours: torch.Tensor,
    transmittance: torch.Tensor = None,
    img_size = (256, 256),
):
    H, W = img_size

    # Do Splatting
    
    ## Project Means
    ##### means_2d = camera.transform_points_screen(means_3d, img_size = (W, H))[:, :2]
    means_2d = camera.transform_points_to_screen(means_3d)[:, :2]
    
    ## Compute Cov
    ##### w2c = camera.get_world_to_view_transform()
    w2c = camera.get_w2c_transform()
    ##### J = compute_jacobian(means_3d, camera.focal_length.flatten(), w2c, img_size)
    J = compute_jacobian(means_3d, (camera.focal_length, camera.focal_length), w2c, img_size)
    ##### W_mat = w2c.get_matrix()[:, :3, :3]
    W_mat = w2c.matrix[None, :3, :3]
    ### 各向异性的时候的计算
    S = torch.diag_embed(scales)  # (N, 3, 3)
    normalized_quat = torch.where(quats[..., :1] < 0, -quats, quats)
    R = quat_to_mat(normalized_quat)
    cov_3d = R @ S @ (R @ S).transpose(1, 2)
    cov_2d = J @ W_mat @ cov_3d @ W_mat.transpose(1, 2) @ J.transpose(1, 2)
    ### ???
    cov_2d[:, 0, 0] += 0.3
    cov_2d[:, 1, 1] += 0.3

    ## Compute Alphas
    xs, ys = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    points_2D = torch.stack((xs.flatten(), ys.flatten()), dim = 1).to(means_3d.device)  # (H*W, 2)

    points_2D = points_2D.unsqueeze(0)  # (1, H*W, 2)
    means_2d = means_2d.unsqueeze(1)  # (N, 1, 2)

    cov_2d_inv = invert_cov_2D(cov_2d)
    diff = points_2D - means_2d
    product = diff @ cov_2d_inv
    power = -0.5 * torch.sum(product * diff, dim = 2)
    exp_power = torch.exp(power)
    exp_power[exp_power >= 1.0] = 0.0
    
    alphas = opacities.unsqueeze(1) * exp_power
    alphas = torch.reshape(alphas, (-1, H, W))  # (N, H, W)
    # Post processing for numerical stability
    alphas = torch.minimum(alphas, torch.full_like(alphas, 0.99))
    alphas = torch.where(alphas < 1 / 255.0, 0.0, alphas)
    
    # Transmittance
    if (transmittance is None):
        start_trans = torch.ones((1, H, W), device=alphas.device)
    else:
        start_trans = transmittance
    one_minus_alphas = torch.concat([
        start_trans, 1.0 - alphas + 1e-10
    ], dim = 0)
    transmittance = torch.cumprod(one_minus_alphas, dim = 0)[1:, ...]
    transmittance = torch.where(transmittance < 1e-4, 0.0, transmittance)  # (N, H, W)


    z_vals = z_vals[:, None, None, None]  # (N, 1, 1, 1)
    alphas = alphas[..., None]  # (N, H, W, 1)
    colours = colours[:, None, None, :]  # (N, 1, 1, 3)
    transmittance = transmittance[..., None]  # (N, H, W, 1)

    weights = alphas * transmittance  # (N, H, W, 1)
    
    rgb_map = torch.sum(weights * colours, dim = 0)  # (H, W, 3)
    depth_map = torch.sum(weights * z_vals, dim = 0)  # (H, W, 1)
    acc_map = torch.sum(weights, dim = 0)
    
    final_transmittance = transmittance[-1, ..., 0].unsqueeze(0)  # (1, H, W)
    
    return rgb_map, depth_map, acc_map, final_transmittance

def render(
    camera: PerspectiveCamera,
    gaussians: GaussianModel,
    batch_size =4096,
    bg_color = (0.0, 0.0, 0.0),
):
    ##### img_size = camera.image_size.type(torch.int32)[0]
    img_size = torch.tensor(camera.resolution, device=gaussians.means.device, dtype=torch.int32)
    # Compute Depth values
    ##### z_vals = camera.get_world_to_view_transform().transform_points(gaussians.means)[..., 2]
    z_vals = camera.get_w2c_transform().transform_points(gaussians.means)[..., 2]
    # Sort by depth (far to near)
    sorted_z_vals, sorted_indices = torch.sort(z_vals, descending=False)
    sorted_indices = sorted_indices[sorted_z_vals > 0]

    
    # Get valid gaussians
    means_3d = gaussians.means[sorted_indices]
    colours = gaussians.colours[sorted_indices]

    pre_act_quats = gaussians.pre_act_quats[sorted_indices]
    pre_act_scales = gaussians.pre_act_scales[sorted_indices]
    pre_act_opacities = gaussians.pre_act_opacities[sorted_indices]
    z_vals = z_vals[sorted_indices]
    
    # Activate
    scales = torch.exp(pre_act_scales)
    quats = torch.nn.functional.normalize(pre_act_quats)
    opacities = torch.sigmoid(pre_act_opacities)
    
    num_of_gaussians = gaussians.num_of_gaussians
    num_of_batch = math.ceil(num_of_gaussians / batch_size)
    
    final_rgb = torch.zeros((img_size[0], img_size[1], 3), device=means_3d.device)
    final_depth = torch.zeros((img_size[0], img_size[1], 1), device=means_3d.device)
    final_acc = torch.zeros((img_size[0], img_size[1], 1), device=means_3d.device)
    
    transmittance = torch.ones((1, img_size[0], img_size[1]), device=means_3d.device)
    
    for idx in trange(num_of_batch):
        start_idx = idx * batch_size
        end_idx = min((idx + 1) * batch_size, num_of_gaussians)
        batch_means_3d = means_3d[start_idx:end_idx]
        batch_scales = scales[start_idx:end_idx]
        batch_quats = quats[start_idx:end_idx]
        batch_opacities = opacities[start_idx:end_idx]
        batch_colours = colours[start_idx:end_idx]
        batch_z_vals = z_vals[start_idx:end_idx]
        
        rgb_map, depth_map, acc_map, transmittance = splat(
            camera,
            batch_means_3d, batch_z_vals, batch_quats,
            batch_scales, batch_opacities, batch_colours,
            transmittance = transmittance,
            img_size = img_size
        )
        
        final_rgb += rgb_map
        final_depth += depth_map
        final_acc += acc_map
        
    
    final_rgb = final_rgb * final_acc + (1.0 - final_acc) * torch.tensor(bg_color, device=final_rgb.device)

    
    final_depth = torch.where(final_acc > 0.0, final_depth / final_acc, torch.zeros_like(final_depth))
    
    return final_rgb, final_depth, final_acc