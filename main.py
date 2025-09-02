import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from Cameras import create_camera
from Gaussian import GaussianModel
from Renderer import render

if __name__ == "__main__":
    GS_PATH = os.path.join("data", "chair", "point_cloud", "iteration_30000", "point_cloud.ply")
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    gaussian = GaussianModel(init_type="Gaussians", path=GS_PATH, device = DEVICE)
    dist, azim, elev = 6.0, 45, 0.0
    camera = create_camera(
        dist, azim, elev, dim = 256, device = DEVICE
    )
    
    img_raw, depth, acc = render(camera, gaussian, bg_color = (1.0, 1.0, 1.0))
    
    img = img_raw.cpu().numpy()
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.axis("off")
    plt.savefig("rendered_image.png")
    plt.show()