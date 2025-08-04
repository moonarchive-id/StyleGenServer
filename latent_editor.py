import typing as tp
import numpy as np
import torch
import os

class LatentEditor:
    def __init__(self, boundary_dir: str = "interfacegan_boundaries"):
        self.boundary_dir = boundary_dir
        self.loaded = {}

    def load_boundary(self, name: str) -> np.ndarray:
        if name not in self.loaded:
            path = os.path.join(self.boundary_dir, f"stylegan_ffhq_{name}_w_boundary.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Boundary '{name}' tidak ditemukan di {path}")
            self.loaded[name] = np.load(path)
        return self.loaded[name]

    def edit(
        self,
        latent: torch.Tensor,
        direction: str,
        alpha: float,
        layers: tp.Optional[list[int]] = None
    ) -> torch.Tensor:
        boundary = self.load_boundary(direction)
        boundary_tensor = torch.from_numpy(boundary).to(latent.device).float()
        
        if boundary_tensor.ndim == 1: 
            boundary_tensor = boundary_tensor.unsqueeze(0)

        if latent.ndim == 3 and latent.shape[1] == 18:
            boundary_tensor_expanded = boundary_tensor.expand(latent.shape[0], latent.shape[1], -1)
            if layers:
                for l in layers:
                    if l < latent.shape[1]:
                        latent[:, l, :] += alpha * boundary_tensor_expanded[:, l, :]
            else:
                latent += alpha * boundary_tensor_expanded
        else:
            raise ValueError(f"Latent shape tidak dikenali: {latent.shape}")
        return latent