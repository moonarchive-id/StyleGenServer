import argparse
import typing as tp
from collections import defaultdict
from functools import wraps
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.io import read_image, ImageReadMode

from models.Alignment import Alignment
from models.Blending import Blending
from models.Embedding import Embedding
from models.Net import Net
from utils.image_utils import equal_replacer
from utils.seed import seed_setter
from utils.shape_predictor import align_face
from utils.time import bench_session
from latent_editor import LatentEditor
from models.stylegan2.model import Generator as StyleGAN2Generator

# Type aliases
TImage = tp.TypeVar('TImage', torch.Tensor, Image.Image, np.ndarray)
TPath = tp.TypeVar('TPath', Path, str)
TReturn = tp.TypeVar('TReturn', torch.Tensor, tuple[torch.Tensor, ...])


class HairFast:
    def __init__(self, args):
        self.args = args
        self.net = Net(self.args)
        self.embed = Embedding(args, net=self.net)
        self.align = Alignment(args, self.embed.get_e4e_embed, net=self.net)
        self.blend = Blending(args, net=self.net)
        self.editor = LatentEditor()
        self.gan = StyleGAN2Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(args.device)
        ckpt = torch.load(args.ckpt, map_location=args.device)
        self.gan.load_state_dict(ckpt["g_ema"], strict=False)
        self.gan.eval()

        # Layer mapping for each edit type
        self.boundary_layers = {
            'smile': list(range(8, 12)),
            'age': list(range(0, 6)),
            'gender': list(range(0, 6)),
            'pose': list(range(0, 6)),
            'glasses': list(range(10, 14))
        }

    def edit_latent(self, latent, smile=0.0, age=0.0, gender=0.0, pose=0.0, glasses=0.0):
        latent_tensor = None

        if isinstance(latent, dict):
            print(f"[DEBUG] Keys dalam latent dict: {list(latent.keys())}")
            for key in ['latent', 'latent_input', 'w', 'codes', 'W']:
                if key in latent:
                    latent_tensor = latent[key]
                    print(f"[DEBUG] Menggunakan latent dari key: {key}")
                    break
            if latent_tensor is None:
                raise KeyError("Tidak ditemukan key valid ('latent', 'latent_input', 'W', dll).")
        elif isinstance(latent, torch.Tensor):
            latent_tensor = latent
        else:
            raise TypeError(f"Tipe latent tidak dikenali: {type(latent)}")

        print(f"[DEBUG] Latent shape sebelum edit: {latent_tensor.shape}")

        # Apply edits
        if smile != 0.0:
            if latent_tensor.ndim == 3 and latent_tensor.shape[1] >= 12:
                latent_tensor = self.editor.edit(latent_tensor, 'smile', smile, layers=self.boundary_layers['smile'])
            else:
                latent_tensor = self.editor.edit(latent_tensor, 'smile', smile)
            print(f"[DEBUG] Latent setelah edit (smile): {latent_tensor[0, 0, :5]}")
        if age != 0.0:
            latent_tensor = self.editor.edit(latent_tensor, 'age', age, layers=self.boundary_layers['age'])
            print(f"[DEBUG] Latent setelah edit (age): {latent_tensor[0, 0, :5]}")
        if gender != 0.0:
            latent_tensor = self.editor.edit(latent_tensor, 'gender', gender, layers=self.boundary_layers['gender'])
            print(f"[DEBUG] Latent setelah edit (gender): {latent_tensor[0, 0, :5]}")
        if pose != 0.0:
            latent_tensor = self.editor.edit(latent_tensor, 'pose', pose, layers=self.boundary_layers['pose'])
            print(f"[DEBUG] Latent setelah edit (pose): {latent_tensor[0, 0, :5]}")
        if glasses != 0.0:
            if latent_tensor.ndim == 3 and latent_tensor.shape[1] >= 14:
                latent_tensor = self.editor.edit(latent_tensor, 'glasses', glasses, layers=self.boundary_layers['glasses'])
            else:
                latent_tensor = self.editor.edit(latent_tensor, 'glasses', glasses)
            print(f"[DEBUG] Latent setelah edit (glasses): {latent_tensor[0, 0, :5]}")

        # Pastikan input ke .style adalah W [1, 512]
        if latent_tensor.ndim == 3 and latent_tensor.shape[1] == 18:
            w = latent_tensor.mean(dim=1)  # [1, 512]
        elif latent_tensor.ndim == 2 and latent_tensor.shape[1] == 512:
            w = latent_tensor
        else:
            raise ValueError(f"[ERROR] Format latent tidak dikenali: {latent_tensor.shape}")

        # .style expects [1, 512]
        style_tensor = self.net.generator.style(w)  # [1, 512]

        # Expand jadi [1, 18, 512] agar cocok downstream
        style_tensor = style_tensor.unsqueeze(1).repeat(1, 18, 1)
        print(f"[DEBUG] Style tensor setelah expand: {style_tensor.shape}")

        if isinstance(latent, dict):
            latent['W'] = latent_tensor
            latent['S'] = style_tensor
            return latent

        return latent_tensor


    @seed_setter
    @bench_session
    def __swap_from_tensors(self, face: torch.Tensor, shape: torch.Tensor, color: torch.Tensor, **kwargs) -> torch.Tensor:
        images_to_name = defaultdict(list)
        for image, name in zip((face, shape, color), ('face', 'shape', 'color')):
            images_to_name[image].append(name)

        name_to_embed = self.embed.embedding_images(images_to_name, **kwargs)

        if 'face' in name_to_embed:
            latent = name_to_embed['face']
            smile = kwargs.get('smile', 0.0)
            age = kwargs.get('age', 0.0)
            gender = kwargs.get('gender', 0.0)
            pose = kwargs.get('pose', 0.0)
            glasses = kwargs.get('glasses', 0.0)
            name_to_embed['face'] = self.edit_latent(latent, smile, age, gender, pose, glasses)

        align_shape = self.align.align_images('face', 'shape', name_to_embed, **kwargs)
        align_color = self.align.shape_module('face', 'color', name_to_embed, **kwargs) if shape is not color else align_shape

        final_image = self.blend.blend_images(align_shape, align_color, name_to_embed, **kwargs)
        return final_image

    def swap(self, face_img: TImage | TPath, shape_img: TImage | TPath, color_img: TImage | TPath,
             benchmark=False, align=False, seed=None, exp_name=None, **kwargs) -> TReturn:
        images: list[torch.Tensor] = []
        path_to_images: dict[TPath, torch.Tensor] = {}

        for img in (face_img, shape_img, color_img):
            if isinstance(img, (torch.Tensor, Image.Image, np.ndarray)):
                if not isinstance(img, torch.Tensor):
                    img = F.to_tensor(img)
                img = F.resize(img, [1024, 1024])
            elif isinstance(img, (Path, str)):
                path_img = img
                if path_img not in path_to_images:
                    path_to_images[path_img] = read_image(str(path_img), mode=ImageReadMode.RGB)
                img = path_to_images[path_img]
                img = F.resize(img, [1024, 1024])
            else:
                raise TypeError(f'Unsupported image format {type(img)}')

            images.append(img)

        if align:
            images = align_face(images)
        images = equal_replacer(images)

        final_image = self.__swap_from_tensors(*images, seed=seed, benchmark=benchmark, exp_name=exp_name, **kwargs)

        if align:
            return final_image, *images
        return final_image

    @wraps(swap)
    def __call__(self, *args, **kwargs):
        return self.swap(*args, **kwargs)


def get_parser():
    parser = argparse.ArgumentParser(description='HairFast')

    parser.add_argument('--save_all_dir', type=Path, default=Path('output'), help='Directory to save the latent codes and inversion images')
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--ckpt', type=str, default="pretrained_models/StyleGAN/ffhq.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--latent', type=int, default=512)
    parser.add_argument('--n_mlp', type=int, default=8)

    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--save_all', action='store_true')

    parser.add_argument('--mixing', type=float, default=0.95)
    parser.add_argument('--smooth', type=int, default=5)
    parser.add_argument('--rotate_checkpoint', type=str, default='pretrained_models/Rotate/rotate_best.pth')
    parser.add_argument('--blending_checkpoint', type=str, default='pretrained_models/Blending/checkpoint.pth')
    parser.add_argument('--pp_checkpoint', type=str, default='pretrained_models/PostProcess/pp_model.pth')

    parser.add_argument('--smile', type=float, default=0.0)
    parser.add_argument('--age', type=float, default=0.0)
    parser.add_argument('--gender', type=float, default=0.0)
    parser.add_argument('--pose', type=float, default=0.0)
    parser.add_argument('--glasses', type=float, default=0.0)

    return parser


if __name__ == '__main__':
    model_args = get_parser()
    args = model_args.parse_args()
    hair_fast = HairFast(args)