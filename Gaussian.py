import torch
import numpy as np
from plyfile import PlyData

from utils import load_gaussians_from_ply

class GaussianModel:
    means: torch.Tensor # (N, 3), 表示 Gaussian 的中心位置
    pre_act_quats: torch.Tensor # (N, 4), 表示 Gaussian 的旋转，使用四元数表示
    pre_act_scales: torch.Tensor # (N, 3), 表示 Gaussian 的缩放
    pre_act_opacities: torch.Tensor # (N, 1), 表示 Gaussian 的不透明度
    colours: torch.Tensor # (N, 3), 表示 Gaussian 的颜色
    num_of_gaussians: int # Gaussian 的数量
    is_isotropic: bool # 是否各向同性
    
    def __init__(self, init_type: str = "Gaussians", path: str = None, device: str = "cpu"):
        if (init_type == "Gaussians"):
            if (path is None):
                raise ValueError("Path must be provided for Gaussian initialization.")
            
            raw = load_gaussians_from_ply(path)
            
            self.means = torch.tensor(raw["xyz"], device=device)
            self.pre_act_quats = torch.tensor(raw["rot"], device=device)
            self.pre_act_scales = torch.tensor(raw["scale"], device=device)
            self.pre_act_opacities = torch.tensor(raw["opacity"], device=device).squeeze()
            self.colours = torch.tensor(raw["dc_colours"], device=device)
            self.is_isotropic = False
            self.num_of_gaussians = self.means.shape[0]
        else:
            raise NotImplementedError(f"Initialization type {init_type} not implemented.")
        
        # Center the Gaussians around the origin
        self.means -= self.means.mean(dim=0, keepdim=True)
        
if __name__ == "__main__":
    gaussian = GaussianModel(init_type="Gaussians", path="./data/chair/point_cloud/iteration_30000/point_cloud.ply")
    
    