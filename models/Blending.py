import torch
from torch import nn

from models.Encoders import ClipBlendingModel, PostProcessModel
from models.Net import Net
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.save_utils import save_gen_image, save_latents


class Blending(nn.Module):
    """
    Module for transferring the desired hair color and post processing
    """

    def __init__(self, opts, net=None):
        super().__init__()
        self.opts = opts
        self.net = net if net is not None else Net(self.opts)

        # Load blending encoder
        blending_checkpoint = torch.load(self.opts.blending_checkpoint, map_location=self.opts.device)
        self.blending_encoder = ClipBlendingModel(blending_checkpoint.get('clip', "ViT-B/32"))
        self.blending_encoder.load_state_dict(blending_checkpoint['model_state_dict'], strict=False)
        self.blending_encoder.to(self.opts.device).eval()

        # Load post process
        self.post_process = PostProcessModel().to(self.opts.device).eval()
        self.post_process.load_state_dict(
            torch.load(self.opts.pp_checkpoint, map_location=self.opts.device)['model_state_dict']
        )

        self.dilate_erosion = DilateErosion(dilate_erosion=self.opts.smooth, device=self.opts.device)
        self.downsample_256 = BicubicDownSample(factor=4)

    @torch.inference_mode()
    def blend_images(self, align_shape, align_color, name_to_embed, **kwargs):
        I_1 = name_to_embed['face']['image_norm_256']
        I_2 = name_to_embed['shape']['image_norm_256']
        I_3 = name_to_embed['color']['image_norm_256']

        # Buat mask untuk blending
        mask_de = self.dilate_erosion.hair_from_mask(
            torch.cat([name_to_embed[x]['mask'] for x in ['face', 'color']], dim=0)
        )
        HM_1D, _ = mask_de[0][0].unsqueeze(0), mask_de[1][0].unsqueeze(0)
        HM_3D, HM_3E = mask_de[0][1].unsqueeze(0), mask_de[1][1].unsqueeze(0)

        latent_S_1 = name_to_embed['face']['S']
        latent_S_3 = name_to_embed['color']['S']
        latent_F_align = align_shape['latent_F_align']
        HM_X = align_color['HM_X']

        HM_XD, _ = self.dilate_erosion.mask(HM_X)
        target_mask = (1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)

        # Blending jika wajah, warna, dan bentuk berbeda
        if not (torch.equal(I_1, I_2) and torch.equal(I_1, I_3)):
            print("[DEBUG] latent_S_1[:, 6:] shape:", latent_S_1[:, 6:].shape)
            print("[DEBUG] latent_S_3[:, 6:] shape:", latent_S_3[:, 6:].shape)
            print("[DEBUG] latent_S_1[:, 6:18, :].shape =", latent_S_1[:, 6:18, :].shape)  # Harus [1, 12, 512]

            S_blend_6_18 = self.blending_encoder(
                latent_S_1[:, 6:18, :],
                latent_S_3[:, 6:18, :],
                I_1,  # target_face
                kwargs.get("hair_color", I_3)  # fallback ke I_3 jika hair_color tidak diberikan
            )
            S_blend = torch.cat((latent_S_1[:, :6, :], S_blend_6_18), dim=1)
        else:
            S_blend = latent_S_1

        # Generate blended image
        I_blend, _ = self.net.generator(
            [S_blend], input_is_latent=True, return_latents=False,
            start_layer=4, end_layer=8, layer_in=latent_F_align
        )
        I_blend_256 = self.downsample_256(I_blend)

        # Post-process
        S_final, F_final = self.post_process(I_1, I_blend_256)
        I_final, _ = self.net.generator(
            [S_final], input_is_latent=True, return_latents=False,
            start_layer=5, end_layer=8, layer_in=F_final
        )

        # Optional: Save semua intermediate jika opsi aktif
        if self.opts.save_all:
            exp_name = kwargs.get('exp_name', '')
            output_dir = self.opts.save_all_dir / exp_name

            save_gen_image(output_dir, 'Blending', 'blending.png', I_blend)
            save_latents(output_dir, 'Blending', 'blending.npz', S_blend=S_blend)

            save_gen_image(output_dir, 'Final', 'final.png', I_final)
            save_latents(output_dir, 'Final', 'final.npz', S_final=S_final, F_final=F_final)

        final_image = ((I_final[0] + 1) / 2).clip(0, 1)
        return final_image
