import torch
import math
from typing import Tuple


from pytorch3d.renderer.cameras import PerspectiveCameras, look_at_view_transform
import torch

# Camera Functions and Utilities
def look_at(
    position: torch.Tensor,
    target: torch.Tensor,
    ref_up: torch.Tensor = torch.tensor([0, -1, 0], dtype=torch.float32),
    device: str = "cpu"
) -> torch.Tensor:
    assert position.shape == (3,)
    assert target.shape == (3,)
    assert ref_up.shape == (3,)
    
    z_axis = torch.nn.functional.normalize(
        target - position,
        dim = 0,
        eps = 1e-5
    )
    x_axis = torch.nn.functional.normalize(
        torch.cross(ref_up, z_axis),
        dim = 0,
        eps = 1e-5
    )
    y_axis = torch.nn.functional.normalize(
        torch.cross(z_axis, x_axis),
        dim = 0,
        eps = 1e-5
    )
    
    # Check the Gimbal Lock
    x_closeness = torch.isclose(x_axis, torch.zeros(3, device = device), atol=1e-5)
    if (x_closeness.any()):
        new_x_axis = torch.nn.functional.normalize(
            torch.cross(y_axis, z_axis),
            dim = 0,
            eps = 1e-5
        )
        x_axis = torch.where(x_closeness, new_x_axis, x_axis)
    
    # Generate Rotation Matrix
    R = torch.stack([x_axis, y_axis, z_axis], dim=0).transpose(0, 1)
    return R

def compute_position_from_sh(
    dist: float, elev: float, azim: float, 
    is_degree: bool = True, device: str = "cpu"
) -> torch.Tensor:
    if (is_degree):
        elev_rad = torch.deg2rad(torch.tensor(elev, dtype = torch.float32, device = device))
        azim_rad = torch.deg2rad(torch.tensor(azim, dtype = torch.float32, device = device))
    else:
        elev_rad = torch.tensor(elev, dtype=torch.float32, device = device)
        azim_rad = torch.tensor(azim, dtype=torch.float32, device = device)
    x = dist * torch.cos(elev_rad) * torch.sin(azim_rad)
    y = dist * torch.sin(elev_rad)
    z = dist * torch.cos(elev_rad) * torch.cos(azim_rad)
    position = torch.stack([x, y, z], dim = 0)
    return position

def get_RT(
    dist: float = 1.0, elev: float = 0.0, azim: float = 0.0, 
    camera_position: torch.Tensor = None, 
    is_degree: bool = True,
    target: torch.Tensor = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
    ref_up: torch.Tensor = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
    device: str = "cpu"
):
    target = target.to(device)
    ref_up = ref_up.to(device)
    if (camera_position is not None):
        position = camera_position.to(device)
    else:
        position = compute_position_from_sh(dist, elev, azim, is_degree = is_degree, device = device) + target
    
    R = look_at(position, target, ref_up, device = device)
    T = -torch.matmul(R.transpose(0, 1), position[:, None])[:, 0]
    return R, T


class Transform:
    def __init__(self, transform_matrix: torch.Tensor, device: str = "cpu"):
        """
        初始化变换对象
        
        参数:
            transform_matrix: 变换矩阵，形状为 (D+1, D+1)，其中 D 是维度 (2, 3 或 4)
        """
        # 检查矩阵形状
        dim = transform_matrix.shape[0] - 1
        if dim not in [2, 3, 4]:
            raise ValueError(f"变换矩阵必须是 3×3, 4×4 或 5×5，实际为 {transform_matrix.shape}")
        if transform_matrix.shape != (dim+1, dim+1):
            raise ValueError(f"变换矩阵形状应为 ({dim+1}, {dim+1})，实际为 {transform_matrix.shape}")
        
        self.matrix = transform_matrix
        self.dim = dim  # 存储维度信息
        self.device = device
        self.matrix = self.matrix.to(device)
    
    def transform_points(self, points: torch.Tensor) -> torch.Tensor:
        # 检查点集维度
        if points.shape[-1] != self.dim:
            raise ValueError(f"点集维度应为 {self.dim}，实际为 {points.shape[-1]}")
        
        # 保存原始形状以便恢复
        original_shape = points.shape
        
        # 展平批次维度
        points = points.view(-1, self.dim)
        
        # 转换为齐次坐标
        ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
        points_h = torch.cat([points, ones], dim=-1)
        
        # 应用变换
        transformed_points_h = points_h @ self.matrix.T
        
        # 分离坐标和 w 分量
        transformed_points = transformed_points_h[..., :self.dim]
        w = transformed_points_h[..., self.dim:]
        
        # 避免除以零
        w_zero = torch.abs(w) < 1e-10
        w[w_zero] = 1e-10 * torch.sign(w[w_zero] + 1e-10)

        # 归一化齐次坐标
        transformed_points = transformed_points / w
        
        # 恢复原始形状
        return transformed_points.view(original_shape)

    def inverse(self):
        inv_matrix = torch.linalg.inv(self.matrix)
        return Transform(inv_matrix)


class PerspectiveCamera:
    # Extrinsics
    Rotate_Matrix: torch.Tensor # (3, 3)
    Translate_Vector: torch.Tensor # (3, )
    
    # Intrinsics 
    resolution: Tuple[int, int] # (Height, Width)
    focal_length: float
    principal_point: Tuple[float, float] # (px, py)
    z_near: float
    z_far: float
    
    device: str = "cpu"
    def __init__(self, camera_configs: dict):
        # Check the type of configs.
        assert camera_configs["Camera_Type"] == "Perspective"
        cam_cfg_type: dict = camera_configs["Config_Type"]
        Extrinsic_Config_Raw: dict = camera_configs["Extrinsic"]
        Intrinsic_Config_Raw: dict = camera_configs["Intrinsic"]
        device = camera_configs.get("Device", "cpu")
        
        # Parse Configs
        Extrinsic_Config = {
            "R": None,
            "T": None
        }
        Intrinsic_Config = {
            "Focal_Length": None,
            "Principle_Point": None,
            "Resolution": None,
            "Z_near_far": None
        }
        if (cam_cfg_type == "NeRF"):
            raise NotImplementedError("Not Implement NeRF yet")
        elif (cam_cfg_type == "PyTorch3D"):
            Extrinsic_Config["R"] = Extrinsic_Config_Raw["R"]
            Extrinsic_Config["T"] = Extrinsic_Config_Raw["T"]
            
            dimension = Intrinsic_Config_Raw["Dimension"]
            Intrinsic_Config["Focal_Length"] = Intrinsic_Config_Raw.get(
                "Focal_Length", 5.0 * dimension / 2.0
            )
            Intrinsic_Config["Principle_Point"] = Intrinsic_Config_Raw.get(
                "Principle_Point", (dimension / 2, dimension / 2)
            )
            Intrinsic_Config["Resolution"] = Intrinsic_Config_Raw.get(
                "Resolution", (dimension, dimension)
            )
            Intrinsic_Config["Z_near_far"] = Intrinsic_Config_Raw.get(
                "Z_near_far", (0.1, 10.0)
            )
        elif (cam_cfg_type == "Customize"):
            # Given the distance, azimuth, elevation
            dist = Extrinsic_Config_Raw["Distance"]
            elev = Extrinsic_Config_Raw["Elevation"]
            azim = Extrinsic_Config_Raw["Azimuth"]
            
            R, T = get_RT(dist, elev, azim, device = device)
            Extrinsic_Config["R"] = R
            Extrinsic_Config["T"] = T
            
            dimension = Intrinsic_Config_Raw["Dimension"]
            Intrinsic_Config["Focal_Length"] = Intrinsic_Config_Raw.get(
                "Focal_Length", 5.0 * dimension / 2.0
            )
            Intrinsic_Config["Principle_Point"] = Intrinsic_Config_Raw.get(
                "Principle_Point", (dimension / 2, dimension / 2)
            )
            Intrinsic_Config["Resolution"] = Intrinsic_Config_Raw.get(
                "Resolution", (dimension, dimension)
            )
            Intrinsic_Config["Z_near_far"] = Intrinsic_Config_Raw.get(
                "Z_near_far", (0.1, 10.0)
            )
            
        else:
            raise ValueError(f"Unknown camera config type: {camera_configs['Config_Type']}")
        
        
        self.Rotate_Matrix = Extrinsic_Config["R"]
        self.Translate_Vector = Extrinsic_Config["T"]
        
        self.focal_length = Intrinsic_Config["Focal_Length"]
        self.principal_point = Intrinsic_Config["Principle_Point"]
        self.resolution = Intrinsic_Config["Resolution"]
        z_near_far = Intrinsic_Config["Z_near_far"]
        self.z_near, self.z_far = z_near_far
        
        self.device = device
        self.Rotate_Matrix = self.Rotate_Matrix.to(device)
        self.Translate_Vector = self.Translate_Vector.to(device)
        
    
    def get_w2c_transform_matrix(self):
        R = self.Rotate_Matrix
        T = self.Translate_Vector
        RT = torch.eye(4, device=R.device)
        RT[:3, :3] = R
        RT[:3, 3] = T
        # Camera Position
        
        return RT
    def get_w2c_transform(self):
        RT = self.get_w2c_transform_matrix()
        return Transform(RT, device = self.device)
    def get_c2w_transform_matrix(self):
        R = self.Rotate_Matrix
        T = self.Translate_Vector
        R_inv = R.T
        T_inv = -R_inv @ T
        RT_inv = torch.eye(4, device=R.device)
        RT_inv[:3, :3] = R_inv
        RT_inv[:3, 3] = T_inv
        return RT_inv
    def get_c2w_transform(self):
        RT_inv = self.get_c2w_transform_matrix()
        return Transform(RT_inv, device = self.device)
    
    def get_projection_transform_matrix(self):
        # 使用生成式创建投影矩阵
        K = torch.tensor([
            [self.focal_length, 0, self.principal_point[1], 0],
            [0, self.focal_length, self.principal_point[0], 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], device=self.device)
        return K
    
    def get_projection_transform(self):
        proj_mat = self.get_projection_transform_matrix()
        return Transform(proj_mat, device = self.device)

    def get_ndc_to_screen_transform_matrix(self):
        H, W = self.resolution
        ndc_to_screen_mat = torch.tensor([
            [W / 2.0, 0.0, 0.0, W / 2.0],
            [0.0, H / 2.0, 0.0, H / 2.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0 * W / 2.0, -1.0 * H / 2.0, 0.0, 1.0]
        ], device=self.device)
        return ndc_to_screen_mat
    def get_ndc_to_screen_transform(self):
        mat = self.get_ndc_to_screen_transform_matrix()
        return Transform(mat, device = self.device)

    def transform_points_to_screen(self, points: torch.Tensor) -> torch.Tensor:
        """
        将世界坐标系下的点转换到屏幕坐标系
        
        参数:
            points: 形状为 (N, 3) 的张量，表示 N 个三维点的世界坐标
        
        返回:
            形状为 (N, 3) 的张量，表示 N 个三维点的屏幕坐标
        """
        # 获取各个变换矩阵
        w2c_transform = self.get_w2c_transform()
        proj_transform = self.get_projection_transform()
        
        camera_points = w2c_transform.transform_points(points)
        screen_points = proj_transform.transform_points(camera_points)
        return screen_points
    def transform_points_to_ndc(self, point: torch.Tensor) -> torch.Tensor:
        w2c_transform = self.get_w2c_transform()
        proj_transform = self.get_projection_transform()
        ndc_to_screen = self.get_ndc_to_screen_transform()
        
        full_proj_matrix = w2c_transform.compose(proj_transform)
        inv_ndc = ndc_to_screen.inverse()
        final = full_proj_matrix.compose(inv_ndc)
        
        return final.transform_points(point)

def create_camera(dist: float, azim: float, elev: float, dim: float, up: tuple = (0, -1, 0), device: str = "cpu", cam_source = "PyTorch3D"):
    R, T = get_RT(dist = dist, azim = azim, elev = elev, ref_up = torch.tensor(up, dtype=torch.float32), device = device)
    print(R.device, T.device)
    camera_config = {
        "Camera_Type": "Perspective",
        "Config_Type": "PyTorch3D",
        "Extrinsic": {
            "R": R.squeeze(),
            "T": T.squeeze()
        },
        "Intrinsic": {
            "Dimension": dim,
            # "Focal_Length": 5.0 * dim / 2.0,
            # "Principle_Point": (dim / 2, dim / 2),
            # "Resolution": (dim, dim)
        },
        "Device": device
    }

    camera = PerspectiveCamera(camera_configs = camera_config)
    return camera

def create_camera_pytorch3d(dist: float, azim: float, elev: float, dim: float, up: tuple = ((0, -1, 0), ), device: str = "cpu", cam_source = "PyTorch3D"):
    R, T = look_at_view_transform(dist = dist, azim = azim, elev = elev, up = up)

    camera = PerspectiveCameras(
        focal_length = 5.0 * dim / 2.0, in_ndc = False,
        principal_point = ((dim / 2, dim / 2), ), # !!! Should be tuple[tuple[float, float]]
        R = R, T = T,
        image_size = ((dim, dim), ),
        device = device
    )
    return camera