import torch
from PIL import Image
import numpy as np

class FluxWanResizeNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "wan_model": (['wan_1.3b', 'wan_14b'], {"default": 'wan_14b'}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("flux_image", "wan_image",)
    FUNCTION = "resize"
    CATEGORY = "Wan Tools"

    def resize(self, image, wan_model):
        img = Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))
        orig_width, orig_height = img.size
        orig_ratio = orig_width / orig_height

        flux_sizes = [(1024, 1024), (1152, 896), (896, 1152), (1280, 768), (768, 1280), (1536, 640), (640, 1536), (1088, 896), (896, 1088)]
        wan_13b_sizes = [(256, 256), (384, 384), (512, 512), (640, 384), (384, 640), (768, 256), (256, 768)]
        wan_14b_sizes = [(448, 448), (672, 672), (896, 896), (1120, 672), (672, 1120), (1344, 448), (448, 1344)]

        def find_closest_size(width, height, sizes, mod):
            min_diff = float('inf')
            best_size = sizes[0]
            for size in sizes:
                w, h = size
                size_ratio = w / h
                diff = abs(orig_ratio - size_ratio)
                if diff < min_diff:
                    min_diff = diff
                    best_size = size
            target_w, target_h = best_size
            if orig_ratio > 1:
                target_h = int(target_w / orig_ratio)
            else:
                target_w = int(target_h * orig_ratio)
            return (target_w - (target_w % mod), target_h - (target_h % mod))

        flux_target = find_closest_size(orig_width, orig_height, flux_sizes, 16)
        wan_target = find_closest_size(orig_width, orig_height, wan_13b_sizes if wan_model == 'wan_1.3b' else wan_14b_sizes, 32)

        flux_img = img.resize(flux_target, Image.LANCZOS)
        wan_img = img.resize(wan_target, Image.LANCZOS)

        flux_np = np.array(flux_img).astype(np.float32) / 255.0
        flux_tensor = torch.from_numpy(flux_np).unsqueeze(0)
        wan_np = np.array(wan_img).astype(np.float32) / 255.0
        wan_tensor = torch.from_numpy(wan_np).unsqueeze(0)
        return (flux_tensor, wan_tensor,)

class SDXLWanResizeNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "wan_model": (['wan_1.3b', 'wan_14b'], {"default": 'wan_14b'}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("sdxl_image", "wan_image",)
    FUNCTION = "resize"
    CATEGORY = "Wan Tools"

    def resize(self, image, wan_model):
        img = Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))
        orig_width, orig_height = img.size
        orig_ratio = orig_width / orig_height

        sdxl_sizes = [(1024, 1024), (1152, 896), (896, 1152), (1216, 832), (832, 1216), (1344, 768), (768, 1344), (1536, 640), (640, 1536)]
        wan_13b_sizes = [(256, 256), (384, 384), (512, 512), (640, 384), (384, 640), (768, 256), (256, 768)]
        wan_14b_sizes = [(448, 448), (672, 672), (896, 896), (1120, 672), (672, 1120), (1344, 448), (448, 1344)]

        def find_closest_size(width, height, sizes, mod):
            min_diff = float('inf')
            best_size = sizes[0]
            for size in sizes:
                w, h = size
                size_ratio = w / h
                diff = abs(orig_ratio - size_ratio)
                if diff < min_diff:
                    min_diff = diff
                    best_size = size
            target_w, target_h = best_size
            if orig_ratio > 1:
                target_h = int(target_w / orig_ratio)
            else:
                target_w = int(target_h * orig_ratio)
            return (target_w - (target_w % mod), target_h - (target_h % mod))

        sdxl_target = find_closest_size(orig_width, orig_height, sdxl_sizes, 64)
        wan_target = find_closest_size(orig_width, orig_height, wan_13b_sizes if wan_model == 'wan_1.3b' else wan_14b_sizes, 32)

        sdxl_img = img.resize(sdxl_target, Image.LANCZOS)
        wan_img = img.resize(wan_target, Image.LANCZOS)

        sdxl_np = np.array(sdxl_img).astype(np.float32) / 255.0
        sdxl_tensor = torch.from_numpy(sdxl_np).unsqueeze(0)
        wan_np = np.array(wan_img).astype(np.float32) / 255.0
        wan_tensor = torch.from_numpy(wan_np).unsqueeze(0)
        return (sdxl_tensor, wan_tensor,)

