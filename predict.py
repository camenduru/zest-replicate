import os, sys
from cog import BasePredictor, Input, Path
from typing import List
sys.path.append('/content/zest_code')
os.chdir('/content/zest_code')

from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from rembg import remove
from PIL import Image
import torch
from ip_adapter import IPAdapterXL
from ip_adapter.utils import register_cross_attention_hook
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import cv2

import DPT.util.io
from torchvision.transforms import Compose
from DPT.dpt.models import DPTDepthModel
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet

def greet(input_image, material_exemplar, transform, model, ip_model):
    img = np.array(input_image)
    img_input = transform({"image": img})["image"]
    with torch.no_grad():
        sample = torch.from_numpy(img_input).unsqueeze(0)
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
    depth_min = prediction.min()
    depth_max = prediction.max()
    bits = 2
    max_val = (2 ** (8 * bits)) - 1
    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (prediction - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(prediction.shape, dtype=depth.dtype)
    out = (out / 256).astype('uint8')
    depth_map = Image.fromarray(out).resize((1024, 1024))
    rm_bg = remove(input_image)
    target_mask = rm_bg.convert("RGB").point(lambda x: 0 if x < 1 else 255).convert('L').convert('RGB')
    mask_target_img = ImageChops.lighter(input_image, target_mask)
    invert_target_mask = ImageChops.invert(target_mask)
    gray_target_image = input_image.convert('L').convert('RGB')
    gray_target_image = ImageEnhance.Brightness(gray_target_image)
    factor = 1.0  # Try adjusting this to get the desired brightness
    gray_target_image = gray_target_image.enhance(factor)
    grayscale_img = ImageChops.darker(gray_target_image, target_mask)
    img_black_mask = ImageChops.darker(input_image, invert_target_mask)
    grayscale_init_img = ImageChops.lighter(img_black_mask, grayscale_img)
    init_img = grayscale_init_img
    ip_image = material_exemplar.resize((1024, 1024))
    init_img = init_img.resize((1024,1024))
    mask = target_mask.resize((1024, 1024))
    num_samples = 1
    images = ip_model.generate(pil_image=ip_image, image=init_img, control_image=depth_map, mask_image=mask, controlnet_conditioning_scale=0.9, num_samples=num_samples, num_inference_steps=30, seed=42)
    return images[0]

class Predictor(BasePredictor):
    def setup(self) -> None:
        base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        image_encoder_path = "models/image_encoder"
        ip_ckpt = "sdxl_models/ip-adapter_sdxl_vit-h.bin"
        controlnet_path = "diffusers/controlnet-depth-sdxl-1.0"
        device = "cuda"
        torch.cuda.empty_cache()
        controlnet = ControlNetModel.from_pretrained(controlnet_path, variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to(device)
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            base_model_path,
            controlnet=controlnet,
            use_safetensors=True,
            torch_dtype=torch.float16,
            add_watermarker=False,
        ).to(device)
        pipe.unet = register_cross_attention_hook(pipe.unet)
        self.ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)
        model_path = "DPT/weights/dpt_hybrid-midas-501f0c75.pt"
        net_w = net_h = 384
        self.model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform = Compose(
                [
                    Resize(
                        net_w,
                        net_h,
                        resize_target=None,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=32,
                        resize_method="minimal",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    normalization,
                    PrepareForNet(),
                ]
            )
        self.model.eval()
    def predict(
        self,
        input_image: Path = Input(description="Input image"),
        material_image: Path = Input(description="Material image"),
    ) -> Path:
        input_image1 = Image.open(input_image)
        input_image2 = Image.open(material_image)
        output_image = greet(input_image1, input_image2, self.transform, self.model, self.ip_model)
        output_image.save('/content/output_image.png')
        return Path('/content/output_image.png')