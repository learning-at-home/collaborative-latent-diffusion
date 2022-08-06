"""
Built using the code from Google Colab [1] assembled by @multimodalart [2].

[1] https://colab.research.google.com/github/multimodalart/latent-diffusion-notebook/blob/main/Latent_Diffusion_LAION_400M_model_text_to_image.ipynb
[2] https://twitter.com/multimodalart
"""

import argparse
import os
import sys
import zipfile
from urllib.request import urlretrieve

sys.path.append("../latent-diffusion")
sys.path.append('../taming-transformers')

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import torch.nn as nn
from hivemind.moe.server.layers.custom_experts import register_expert_class
from hivemind.utils.logging import get_logger
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from taming.models import vqgan

import autokeras as ak
from tensorflow.keras.models import load_model
import numpy as np
from omegaconf import OmegaConf
from tqdm.auto import trange
tqdm_auto_model = __import__("tqdm.auto", fromlist=[None])
sys.modules['tqdm'] = tqdm_auto_model
from einops import rearrange
from torchvision.utils import make_grid
import gc
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import open_clip


logger = get_logger(__name__)


MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model')

COND_NUM_TOKENS = 128
CHANNELS = 3
HEIGHT = WIDTH = 256


def load_safety_model(clip_model):
    """load the safety model"""

    cache_folder = f"{MODEL_PATH}/clip_retrieval/" + clip_model.replace("/", "_")
    if clip_model == "ViT-L/14":
        model_dir = cache_folder + "/clip_autokeras_binary_nsfw"
        dim = 768
    elif clip_model == "ViT-B/32":
        model_dir = cache_folder + "/clip_autokeras_nsfw_b32"
        dim = 512
    else:
        raise ValueError("Unknown clip model")

    if not os.path.exists(model_dir):
        os.makedirs(cache_folder, exist_ok=True)

        path_to_zip_file = cache_folder + "/clip_autokeras_binary_nsfw.zip"
        if clip_model == "ViT-L/14":
            url_model = "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_binary_nsfw.zip"
        elif clip_model == "ViT-B/32":
            url_model = (
                "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_nsfw_b32.zip"
            )
        else:
            raise ValueError("Unknown model {}".format(clip_model))
        urlretrieve(url_model, path_to_zip_file)

        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(cache_folder)

    loaded_model = load_model(model_dir, custom_objects=ak.CUSTOM_OBJECTS)
    loaded_model.predict(np.random.rand(10 ** 3, dim).astype("float32"), batch_size=10 ** 3)

    return loaded_model


def is_unsafe(safety_model, embeddings, threshold=0.5):
    """find unsafe embeddings"""
    nsfw_values = safety_model.predict(embeddings, batch_size=embeddings.shape[0])
    x = np.array([e[0] for e in nsfw_values])
    return True if x > threshold else False


def load_model_from_config(config, ckpt, verbose=False):
    logger.info(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cuda:0")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        logger.error(f"Missing keys: {m}")
    if len(u) > 0 and verbose:
        logger.error(f"Unexpected keys: {u}")

    model = model.half().cuda()
    model.eval()
    return model


def run(model, clip_model, preprocess, safety_model, opt):
    torch.cuda.empty_cache()
    gc.collect()
    if opt.plms:
        opt.ddim_eta = 0
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    prompt = opt.prompt

    all_samples=list()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with model.ema_scope():
                uc = None
                if opt.scale > 0:
                    uc = model.get_learned_conditioning(opt.n_samples * [""])
                for n in trange(opt.n_iter, desc="Sampling"):
                    c = model.get_learned_conditioning(opt.n_samples * [prompt])
                    shape = [4, opt.H//8, opt.W//8]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=opt.n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,
                                                    eta=opt.ddim_eta)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                    # for x_sample in x_samples_ddim:
                    #     x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    #     image_vector = Image.fromarray(x_sample.astype(np.uint8))
                    #     image = preprocess(image_vector).unsqueeze(0)
                    #     with torch.no_grad():
                    #       image_features = clip_model.encode_image(image)
                    #     image_features /= image_features.norm(dim=-1, keepdim=True)
                    #     query = image_features.cpu().detach().numpy().astype("float32")
                    #     unsafe = is_unsafe(safety_model,query,opt.nsfw_threshold)
                    #     if unsafe:
                    #       raise Exception('Potential NSFW content was detected on your outputs. Try again with different prompts. If you feel your prompt was not supposed to give NSFW outputs, this may be due to a bias in the model')
                    all_samples.append(x_samples_ddim)

    grid = torch.stack(all_samples, 0)
    grid = (255. * rearrange(grid, 'n b c h w -> (n b) h w c')).to(torch.uint8).cpu()
    return grid


def get_input_example(batch_size: int, *_unused):
    cond_tokens = torch.empty((batch_size, COND_NUM_TOKENS), dtype=torch.int64)
    initial_images = torch.empty((batch_size, CHANNELS, HEIGHT, WIDTH), dtype=torch.uint8)
    return (cond_tokens, initial_images)


@register_expert_class("DiffusionModule", get_input_example)
class DiffusionModule(nn.Module):
    def __init__(self):
        super().__init__()

        self._safety_model = load_safety_model("ViT-B/32")
        self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        logger.info('Loaded safety model and CLIP')

        config = OmegaConf.load("../latent-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml")
        self._model = load_model_from_config(config, f"{MODEL_PATH}/latent_diffusion_txt2img_f8_large.ckpt")
        self._model = self._model.cuda()
        logger.info('Loaded diffusion model')

    def forward(self, cond_tokens: torch.LongTensor, initial_images: torch.ByteTensor):
        logger.info(f"Running forward pass, "
                    f"cond_tokens.shape={cond_tokens.shape}, initial_images.shape={initial_images.shape}")

        Prompt = "A mecha robot holding a sign that reads: 'Is AI art, art?" #@param{type:"string"}

        args = argparse.Namespace(
            prompt = Prompt,
            ddim_steps=50,
            ddim_eta=0.0,
            n_iter=1,
            W=WIDTH,
            H=HEIGHT,
            n_samples=cond_tokens.shape[0],
            scale=5.0,
            plms=True,
            nsfw_threshold=0.5
        )
        output_images = run(self._model, self._clip_model, self._clip_preprocess, self._safety_model, args)
        assert output_images.dtype == torch.uint8  # note: output dtype is important since it affects bandwidth usage!
        return (output_images,)
