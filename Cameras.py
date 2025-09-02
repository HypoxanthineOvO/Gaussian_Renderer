import torch
import torch.nn as nn
#from utils import look_at_view_transform
from pytorch3d.renderer.cameras import PerspectiveCameras, look_at_view_transform

def create_camera(dist: float, azim: float, elev: float, dim: float, up: tuple = ((0, -1, 0), ), device: str = "cpu", cam_source = "PyTorch3D"):
    R, T = look_at_view_transform(dist = dist, azim = azim, elev = elev, up = up)
    if (cam_source == "PyTorch3D"):

        camera = PerspectiveCameras(
            focal_length = 5.0 * dim / 2.0, in_ndc = False,
            principal_point = ((dim / 2, dim / 2), ), # !!! Should be tuple[tuple[float, float]]
            R = R, T = T,
            image_size = ((dim, dim), ),
            device = device
        )
    elif (cam_source == "Ours"):
        camera = PerspectiveCamera_Ours(
            focal_length = 5.0 * dim / 2.0, in_ndc = False,
            principal_point = (dim / 2, dim / 2),
            R = R, T = T,
            image_size = (dim, dim)
        )
    else:
        raise ValueError(f"Unknown camera source: {cam_source}")
    return camera

class PerspectiveCamera_Ours(nn.Module):
    def __init__(self, focal_length=1.0, principal_point=(0.0, 0.0), 
                 R=torch.eye(3), T=torch.zeros(3), image_size=(256, 256)):
        """
        透视相机类
        
        参数:
            focal_length: 焦距 (标量或二元组)
            principal_point: 主点坐标 (二元组)
            R: 3x3 旋转矩阵
            T: 3D 平移向量
            image_size: 图像尺寸 (高度, 宽度)
        """
        super().__init__()
        