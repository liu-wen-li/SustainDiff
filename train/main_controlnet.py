# coding=utf-8
"""LoRA fine-tuning of Stable Diffusion v1.5 + dual ControlNet for energy-aware
flat layout generation.

Features:
  - one or two ControlNet branches (e.g. room layout + energy field),
  - LoRA adapters on the UNet and/or ControlNet(s),
  - periodic evaluation (FID / LPIPS) and best-model saving.

Run `python main_controlnet.py --help` for the full list of arguments, or see
the README for example commands.
"""
import argparse
import gc
import logging
import math
import os
import random
import shutil
import json
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

import datasets
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger, MultiProcessAdapter
from accelerate.utils import ProjectConfiguration, set_seed, DistributedType
from datasets import load_dataset, load_from_disk
from packaging import version
from peft import LoraConfig, get_peft_model_state_dict

from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    ControlNetModel,
)
from diffusers.loaders import LoraLoaderMixin, PeftAdapterMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr, free_memory
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from evaluation import split_dataset, get_train_transforms, conditional_image_transforms
from training_utils import run_evaluation, add_evaluation_args
from logging_utils import TrainingLogger

# Default caption used when no --metadata_path CSV is provided. Every image in this
# dataset shares the same prompt, so a metadata file is not required.
DEFAULT_CAPTION = "A flat layout of residential buildings. LWL_STYLE_FLAT_LAYOUT"


def unet_lora_save(accelerator, unet, save_dir):
    try:
        unwrapped_unet = accelerator.unwrap_model(unet)
        unet_lora_state_dict = get_peft_model_state_dict(unwrapped_unet)

        StableDiffusionPipeline.save_lora_weights(
            save_directory=save_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )
    except Exception as e:
        raise ValueError(f"Failed to save UNet LoRA weights: {e}")

def controlnet_lora_save(args, accelerator, controlnet, save_dir):
    if not isinstance(args.train_controlnet, list):
        args.train_controlnet = [args.train_controlnet]
    try:
        for i in args.train_controlnet:
            if i < len(controlnet):
                unwrapped_controlnet = accelerator.unwrap_model(controlnet[i])
                controlnet_lora_save_path = os.path.join(save_dir, f"controlnet_lora_{i}")
                os.makedirs(controlnet_lora_save_path, exist_ok=True)
                
                peft_state_dict = get_peft_model_state_dict(unwrapped_controlnet)
                save_file(peft_state_dict, os.path.join(controlnet_lora_save_path, "adapter_model.safetensors"))
                
                unwrapped_controlnet.peft_config['default'].save_pretrained(controlnet_lora_save_path)
    except Exception as e:
        raise ValueError(f"Failed to save ControlNet LoRA weights: {e}")

def save_models(args, accelerator, unet, controlnet, save_dir, epoch=None, metrics=None):
    os.makedirs(save_dir, exist_ok=True)
    
    if args.train_unet:
        unet_lora_save(accelerator, unet, save_dir)

    if args.train_controlnet:
        controlnet_lora_save(args, accelerator, controlnet, save_dir)
    
    # Add metadata
    metadata = {}
    if epoch is not None:
        metadata['epoch'] = str(epoch)
    if metrics is not None:
        for k, v in metrics.items():
            metadata[k] = str(v)


    # Save metadata to a separate JSON file for easy inspection
    metadata_save_path = os.path.join(save_dir, "metadata.json")
    with open(metadata_save_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Saved models to {save_dir}")


# Will error if the minimal version of diffusers is not installed.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")

# Mirror all module logs (including training_utils) to main_log.txt in the working directory.
try:
    _root = logging.getLogger()
    _root.setLevel(logging.INFO)
    log_path = os.path.join(os.getcwd(), "main_log.txt")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    fh.setFormatter(fmt)
    if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == fh.baseFilename for h in _root.handlers):
        _root.addHandler(fh)
    print(f"Logging to file: {log_path}")
except Exception as e:
    print(f"Failed to set up file logging: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning of SD1.5 + ControlNet for flat layout generation."
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True,help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        nargs="+",
        default=None,
        help="Path to pretrained controlnet model(s) or model identifier(s) from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_controlnet",
        type=int,
        nargs="*",
        default=None,
        help="Indices of the ControlNet(s) to train with LoRA, e.g. `0` or `0 1`.",
    )
    parser.add_argument(
        "--train_unet",
        action="store_true",
        help="Whether to train the UNet with LoRA.",
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that the Datasets library can understand."
        ),
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )

    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=100,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )

    parser.add_argument(
        "--eval_mode",
        action="store_true",
        help="Run in evaluation mode instead of training mode.",
    )
    parser.add_argument(
        "--eval_output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--eval_num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate in eval mode.",
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.0,
        help="The scale of the controlnet conditioning. Only used during evaluation.",
    )
    parser.add_argument(
        "--controlnet_conditioning_scale_1",
        type=float,
        default=1.0,
        help="Conditioning scale for the first ControlNet (e.g., seg).",
    )
    parser.add_argument(
        "--controlnet_conditioning_scale_2",
        type=float,
        default=1.0,
        help="Conditioning scale for the second ControlNet (e.g., energy field).",
    )

    parser.add_argument(
        "--use_validation_split",
        action="store_true",
        help="Whether to split dataset into train/validation/test sets.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of data for training set.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio of data for validation set.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Ratio of data for test set.",
    )
    
    parser.add_argument("--val_set_size", type=int, default=None, help="Number of samples for the validation set.")
    parser.add_argument("--test_set_size", type=int, default=None, help="Number of samples for the test set.")
    
    parser.add_argument(
        "--enable_fid",
        action="store_true",
        help="Enable FID calculation during evaluation.",
    )
    parser.add_argument(
        "--enable_lpips",
        action="store_true",
        help="Enable LPIPS calculation during evaluation.",
    )
    parser.add_argument(
        "--use_conditional_embeddings",
        action="store_true",
        default=True,
        help="Whether to use conditional embeddings.",
    )
    # Metric direction. If neither flag is set, it is inferred from the metric name
    # (FID/LPIPS -> lower is better, others -> higher is better).
    parser.add_argument("--lower_is_better", dest="lower_is_better", action="store_true",
                        help="Treat the selected metric as lower-is-better (overrides auto-inference).")
    parser.add_argument("--higher_is_better", dest="lower_is_better", action="store_false",
                        help="Treat the selected metric as higher-is-better (overrides auto-inference).")
    parser.set_defaults(lower_is_better=None)

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.1,
        help="The probability of dropping the conditioning to train the model for classifier-free guidance.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=5.0,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", default=False, help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--unet_rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--controlnet_rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Path to the metadata CSV file for image captions.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the dataset directory.",
    )
    parser.add_argument("--overwrite_cache", action="store_true",help="Overwrite the cached processed dataset.")

    # Add evaluation-related arguments
    add_evaluation_args(parser)

    args = parser.parse_args()

    # When launched with `accelerate`/`torchrun`, prefer the environment-provided local rank.
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        try:
            main_log_path = os.path.join(args.output_dir, "main_log.txt")
            target_logger = logger.logger if isinstance(logger, MultiProcessAdapter) else logger
            need_add = True
            for h in getattr(target_logger, "handlers", []):
                if isinstance(h, logging.FileHandler):
                    base = getattr(h, "baseFilename", None)
                    if base and os.path.abspath(base) == os.path.abspath(main_log_path):
                        need_add = False
                        break
            if need_add:
                fh = logging.FileHandler(main_log_path)
                fh.setLevel(logging.INFO)
                formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                fh.setFormatter(formatter)
                target_logger.addHandler(fh)
            # === New: also attach root logger to the same file so other module logs (e.g., training_utils) write here ===
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            root_need_add = True
            for h in getattr(root_logger, "handlers", []):
                if isinstance(h, logging.FileHandler):
                    base = getattr(h, "baseFilename", None)
                    if base and os.path.abspath(base) == os.path.abspath(main_log_path):
                        root_need_add = False
                        break
            if root_need_add:
                root_fh = logging.FileHandler(main_log_path, encoding="utf-8")
                root_fh.setLevel(logging.INFO)
                root_fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
                root_logger.addHandler(root_fh)
                target_logger.info(f"Root logger attached to {main_log_path}")
        except Exception as e:
            print(f"Failed to set up file logging: {e}")

    # ===== Prominent Device Banner =====
    try:
        dev = accelerator.device
        if accelerator.is_local_main_process:
            if dev.type == "cuda":
                gpu_name = torch.cuda.get_device_name(dev) if torch.cuda.is_available() else "N/A"
                gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
                print("\n" + "=" * 70)
                print(f" USING DEVICE: {dev}  |  GPU: {gpu_name}  |  GPU COUNT: {gpu_count}")
                print(f" MIXED PRECISION: {args.mixed_precision}  |  TF32: {args.allow_tf32}")
                print("=" * 70 + "\n")
            else:
                print("\n" + "=" * 70)
                print(f" USING DEVICE: {dev}  |  CPU MODE")
                print(f" MIXED PRECISION: {args.mixed_precision}")
                print("=" * 70 + "\n")
    except Exception as e:
        if accelerator.is_local_main_process:
            print(f"[Device Info] {e}")
    # ===== End Device Banner =====

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Initialize training logger
    exp_name = Path(args.output_dir).name
    training_logger = TrainingLogger(output_dir=args.output_dir, experiment_name=exp_name)
    training_logger.save_config(args)

    if args.hub_token:
        try:
            from huggingface_hub import login as hf_login
            hf_login(token=args.hub_token)
            logger.info("Authenticated with Hugging Face Hub using provided hub_token.")
        except Exception as e:
            logger.warning(f"Failed to authenticate with Hugging Face Hub token: {e}")



    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler",cache_dir=args.cache_dir)
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, cache_dir=args.cache_dir
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, cache_dir=args.cache_dir
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant, cache_dir=args.cache_dir
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant, cache_dir=args.cache_dir
    )
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError("MPS backend does not support bfloat16 mixed precision")

    # Freeze the unet parameters before adding adapters
    for param in unet.parameters():
        param.requires_grad_(False)

    unet.to(accelerator.device, dtype=weight_dtype)
    # ... existing code ...
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.train_unet and not args.eval_mode:
        unet_lora_config = LoraConfig(
            r=args.unet_rank,
            lora_alpha=args.unet_rank,
            lora_dropout=0.1,
            init_lora_weights="gaussian",
            target_modules=[
                "to_k", "to_q", "to_v", "to_out.0",
                "ff.net.0.proj", "ff.net.2",
            ],
        )

        # Add adapter and make sure the trainable params are in float32.
        unet.add_adapter(unet_lora_config)
    # Move unet, vae and text_encoder to device and cast to weight_dtype


    if args.mixed_precision == "fp16":

        for param in unet.parameters():
            # only upcast trainable parameters (LoRA) into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")



    
    # Load ControlNet model
    controlnet = None
    if args.controlnet_model_name_or_path:
        # Use the cache_dir from args if provided, otherwise default to a local directory
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd15_cache")
        logger.info(f"Using cache directory: {cache_dir}")

        controlnets = []
        if args.controlnet_model_name_or_path:
            for model_path in args.controlnet_model_name_or_path:
                logger.info(f"Loading ControlNet from {model_path}")
                cn = ControlNetModel.from_pretrained(
                    model_path,
                    torch_dtype=weight_dtype,
                    cache_dir=cache_dir
                )
                cn.to(accelerator.device)
                controlnets.append(cn)
        controlnet = controlnets

        if args.train_controlnet and not args.eval_mode:
            # Freeze the base model parameters for LoRA training
            logger.info("Freezing original ControlNet weights.")
            targets = controlnets if isinstance(controlnet, list) else [controlnet]
            for tcn in targets:
                    tcn.requires_grad_(False)

            logger.info("Adding LoRA adapter to ControlNet(s).")

            TARGET_MODULES = [
                        "to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2",
                        # "proj_in", "proj_out", "conv", "conv1", "conv2", "conv_in",
                        # "conv_shortcut", "linear_1", "linear_2",
                        # "time_emb_proj"
                    ]
            controlnet_lora_config = LoraConfig(
                r=args.controlnet_rank,
                lora_alpha=args.controlnet_rank,
                target_modules=TARGET_MODULES,
                lora_dropout=0.1,
                bias="none",
                init_lora_weights="gaussian",
                use_dora=False
            )
            for i, tcn in enumerate(targets):
                if i in args.train_controlnet:
                    if not isinstance(tcn, PeftAdapterMixin):
                        logger.info(f"Dynamically patching ControlNetModel instance at index {i} with PEFT capabilities for training.")
                        tcn.__class__ = type(
                            "PeftInjectedControlNetModel",
                            (tcn.__class__, PeftAdapterMixin),
                            {}
                        )
                    tcn.add_adapter(controlnet_lora_config)

                    # Debug: Log the number of trainable parameters for this specific ControlNet
                    try:
                        cn_trainable = sum(p.numel() for p in tcn.parameters() if p.requires_grad)
                        logger.info(f"Trainable params for ControlNet LoRA at index {i}: {cn_trainable}")
                        if cn_trainable == 0:
                            raise ValueError(f"ControlNet LoRA at index {i} has 0 trainable params. Check target_modules/config.")
                    except Exception as e:
                        logger.warning(f"Failed to count trainable params for ControlNet LoRA at index {i}: {e}")

                    # Make sure the trainable params are in float32
                    if args.mixed_precision == "fp16":
                        for param in tcn.parameters():
                            if param.requires_grad:
                                param.data = param.to(torch.float32)

    lora_layers = []
    if args.train_controlnet:
        # Collect parameters from specified ControlNets to train
        if not isinstance(args.train_controlnet, list):
            args.train_controlnet = [args.train_controlnet]
        for i in args.train_controlnet:
            if i < len(controlnet):
                lora_layers.extend(filter(lambda p: p.requires_grad, controlnet[i].parameters()))
    if args.train_unet:
        lora_layers += list(filter(lambda p: p.requires_grad, unet.parameters()))

    all_trainable_params = lora_layers




    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        all_trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


   

    # Initialize best score tracking for optional best-model saving
    if not hasattr(args, "lower_is_better") or args.lower_is_better is None:
        args.lower_is_better = args.best_model_metric in ["fid", "lpips"]

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            data_dir=args.train_data_dir,
        )
    else:

        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**/*.png")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        metadata_path = args.metadata_path
        df = pd.read_csv(metadata_path) if metadata_path else None

        def add_metadata(example):
            try:
                if example["image"] is None or not hasattr(example["image"], 'filename') or not example["image"].filename:
                    logger.error("ERROR: Invalid image data in example. 'image' field is missing or has no 'filename'.")
                    logger.error(f"Example content: {example}")
                    raise ValueError("Invalid data found in dataset. A file might be corrupt or not an image.")

                file_name = os.path.basename(example["image"].filename)

            except (AttributeError, TypeError) as e:
                logger.error("ERROR: Could not get filename from the following example.")
                logger.error(f"Example content: {example}")
                logger.error(f"This usually means the image file is corrupt or could not be loaded. Original error: {e}")
                raise

            example['image_path'] = example['image'].filename
            if df is None:
                # No metadata CSV: every image uses the same default caption.
                example['text'] = DEFAULT_CAPTION
            else:
                metadata_rows = df[df["file_name"] == file_name]
                if not metadata_rows.empty:
                    example['text'] = metadata_rows.iloc[0]['text']
                else:
                    raise ValueError(f"ERROR: No metadata found for file {file_name} in {metadata_path}.")
            return example
    
    cache_path = os.path.join(args.dataset_path, "processed_dataset_cache")
    if os.path.exists(cache_path) and not args.overwrite_cache:
        logger.info(f"** Loading processed dataset from cache: {cache_path} **")
        dataset = load_from_disk(cache_path)
        logger.info("** Cached dataset loaded successfully! **")
    else:
        logger.info(f"** Processing dataset... (Cache not found or overwrite requested at {cache_path}) **")
        # Use num_proc only if it's a positive integer
        num_map_workers = args.dataloader_num_workers if args.dataloader_num_workers > 0 else None
        dataset = dataset.map(add_metadata, num_proc=num_map_workers)

        logger.info(f"** Saving processed dataset to cache: {cache_path} **")
        dataset.save_to_disk(cache_path)
        logger.info("** Dataset cached successfully! **")

    if args.use_validation_split:
        logger.info("Splitting dataset into train/validation/test sets...")
        splits_dir = os.path.join(args.output_dir, "splits")
        os.makedirs(splits_dir, exist_ok=True)
        
        train_split_path = os.path.join(splits_dir, "train_split.csv")
        val_split_path = os.path.join(splits_dir, "val_split.csv")
        test_split_path = os.path.join(splits_dir, "test_split.csv")

        if os.path.exists(train_split_path) and os.path.exists(val_split_path) and os.path.exists(test_split_path):
            logger.info(f"Loading existing dataset splits from {splits_dir}")
            train_df = pd.read_csv(train_split_path)
            val_df = pd.read_csv(val_split_path)
            test_df = pd.read_csv(test_split_path)
        else:
            logger.info("Creating new dataset splits...")
            train_df, val_df, test_df = split_dataset(
                image_dir=args.train_data_dir,
                metadata_path=args.metadata_path,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                random_state=args.seed,
                output_dir=args.output_dir,
                val_set_size=args.val_set_size,
                test_set_size=args.test_set_size
            )

        # The split CSVs define the partition only (a `file_name` column). Captions and
        # per-plan attributes live in the metadata CSV (`--metadata_path`); attach the
        # `text` column here so the validation/test evaluation has the prompt it needs.
        meta_df = pd.read_csv(args.metadata_path) if args.metadata_path else None

        def _attach_metadata(split_df):
            split_df = split_df[["file_name"]].copy()
            if meta_df is not None:
                split_df = split_df.merge(meta_df[["file_name", "text"]], on="file_name", how="left")
                missing = int(split_df["text"].isna().sum())
                if missing:
                    raise ValueError(
                        f"{missing} file(s) in a split have no matching row in {args.metadata_path}."
                    )
            else:
                split_df["text"] = DEFAULT_CAPTION
            return split_df

        val_df = _attach_metadata(val_df)
        test_df = _attach_metadata(test_df)

        train_files = set(train_df["file_name"].tolist())
        val_files = set(val_df["file_name"].tolist())

        train_dataset = dataset["train"].filter(lambda e: os.path.basename(e["image_path"]) in train_files)
        val_dataset = dataset["train"].filter(lambda e: os.path.basename(e["image_path"]) in val_files)

        logger.info(f"Training on {len(train_dataset)} samples after splitting")
        logger.info(f"Validating on {len(val_dataset)} samples after splitting")

        args._val_df = val_df
        args._test_df = test_df
        args._val_csv_path = val_split_path
        args._test_csv_path = test_split_path
    else:
        train_dataset = dataset["train"]
        val_dataset = None

    # Preprocessing the datasets.
    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)

        if args.conditional_image_dir_1:
            png_files = [f for f in os.listdir(args.conditional_image_dir_1) if f.lower().endswith('.png')][:1]
            if not png_files:
                raise ValueError(f"No PNG files found in conditional_image_dir_1: {args.conditional_image_dir_1}")
            conditional_images = []
            for image_path in examples["image_path"]:
                base_name = os.path.basename(image_path)
                cond_path = os.path.join(args.conditional_image_dir_1, base_name)
                if os.path.exists(cond_path):
                    conditional_images.append(Image.open(cond_path).convert("RGB"))
                else:
                    raise ValueError(f"Conditional image not found for {base_name} in {args.conditional_image_dir_1}")
            examples["conditional_pixel_values_1"] = [conditional_transforms(image) for image in conditional_images]
        if args.conditional_image_dir_2:
            png_files = [f for f in os.listdir(args.conditional_image_dir_2) if f.lower().endswith('.png')][:1]
            if not png_files:
                raise ValueError(f"No PNG files found in conditional_image_dir_2: {args.conditional_image_dir_2}")
            conditional_images_2 = []
            for image_path in examples["image_path"]:
                base_name = os.path.basename(image_path)
                cond_path = os.path.join(args.conditional_image_dir_2, base_name)
                if os.path.exists(cond_path):
                    conditional_images_2.append(Image.open(cond_path).convert("RGB"))
                else:
                    raise ValueError(f"Conditional image not found for {base_name} in {args.conditional_image_dir_2}")
            examples["conditional_pixel_values_2"] = [conditional_transforms(image) for image in conditional_images_2]
        return examples

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        
        # room_counts = torch.stack([torch.tensor(example["room_counts"], dtype=torch.float32) for example in examples])

        
        batch = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
        }

        # Handle conditional pixel values if they exist
        if "conditional_pixel_values_1" in examples[0] and examples[0]["conditional_pixel_values_1"] is not None:
            conditional_pixel_values_1 = torch.stack([example["conditional_pixel_values_1"] for example in examples])
            batch["conditional_pixel_values_1"] = conditional_pixel_values_1.to(memory_format=torch.contiguous_format).float()
        if "conditional_pixel_values_2" in examples[0] and examples[0].get("conditional_pixel_values_2") is not None:
            conditional_pixel_values_2 = torch.stack([example["conditional_pixel_values_2"] for example in examples])
            batch["conditional_pixel_values_2"] = conditional_pixel_values_2.to(memory_format=torch.contiguous_format).float()

        return batch

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names


    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    train_transforms = get_train_transforms(args)
    conditional_transforms = conditional_image_transforms(args)


    with accelerator.main_process_first():
        if args.max_train_samples is not None: 
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = train_dataset.with_transform(preprocess_train)
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    val_dataloader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_dataset = val_dataset.with_transform(preprocess_train)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.train_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.dataloader_num_workers,
        )
    
     # Check if running in evaluation mode
    if args.eval_mode:

        logger.info("***** Running Evaluation in Eval Mode *****")

        # Helper to find the latest checkpoint directory
        def find_latest_checkpoint(output_dir):
            checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))]
            if not checkpoint_dirs:
                return None
            try:
                latest_checkpoint = max(checkpoint_dirs, key=lambda d: int(d.split('-')[1]))
                return os.path.join(output_dir, latest_checkpoint)
            except (ValueError, IndexError):
                return None

        # Define search paths for the model weights in order of priority
        load_paths_to_try = [
            ("best_model", os.path.join(args.output_dir, "best_model")),
            ("last_model", os.path.join(args.output_dir, "last_model")),
        ]
        latest_checkpoint_path = find_latest_checkpoint(args.output_dir)
        if latest_checkpoint_path:
            load_paths_to_try.append(("latest_checkpoint", latest_checkpoint_path))
        load_paths_to_try.append(("output_dir", args.output_dir))

        # Iterate through possible paths to find and load all model components
        for model_type, load_path in load_paths_to_try:
            if os.path.exists(load_path):
                logger.info(f"Loading model for '{model_type}': {load_path}")
                break

        # Run evaluation with the prepared models
        logger.info("***** Running Evaluation *****")
        metrics = run_evaluation(
            args, 
            accelerator, 
            tokenizer, 
            noise_scheduler, 
            vae, 
            text_encoder, 
            unet,
            controlnet,
            load_path, # Pass the actual model path for LoRA loading
            weight_dtype,
            save_images=True, # Save images during evaluation
            dataset_type='test', # Use test set for this evaluation
            conditional_image_dir_1=args.conditional_image_dir_1,
            conditional_image_dir_2=args.conditional_image_dir_2,
            controlnet_conditioning_scale=[args.controlnet_conditioning_scale_1, args.controlnet_conditioning_scale_2]
        ) #room_counts_encoder, 
        return

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    if controlnet:
        unet, optimizer, train_dataloader, lr_scheduler, controlnet = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler, controlnet
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    from logging_utils import init_trackers_and_config
    init_trackers_and_config(accelerator, args)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    best_score = float('inf') if args.lower_is_better else float('-inf') 

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
            path = checkpoints[-1] if checkpoints else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' not found. Starting new training.") 
            args.resume_from_checkpoint = None
            initial_global_step = 0
            raise ValueError(f"Checkpoint '{args.resume_from_checkpoint}' not found. Please check the path.")
        else:
            resume_dir = os.path.join(args.output_dir, path)
            accelerator.print(f"Resuming training from checkpoint {resume_dir}")
            accelerator.load_state(resume_dir, map_location="cpu", strict=False)

            unet_lora_file = os.path.join(resume_dir, "pytorch_lora_weights.safetensors")
            controlnet_lora_dir = os.path.join(resume_dir, "controlnet_lora")
            if args.train_unet:
                try:
                    logger.info(f"Loading UNet LoRA from {unet_lora_file}")
                    unet_lora_state_dict, unet_network_alphas = LoraLoaderMixin.lora_state_dict(unet_lora_file)
                    LoraLoaderMixin.load_lora_into_unet(unet_lora_state_dict, network_alphas=unet_network_alphas, unet=unet)
                except Exception as e:
                    raise ValueError(f"Failed to load UNet LoRA weights: {e}")
            # Load ControlNet LoRA from directory
            if args.train_controlnet:
                try:
                    # Loop through the indices of the models we intend to train
                    if not isinstance(args.train_controlnet, list):
                        args.train_controlnet = [args.train_controlnet]
                    for i in args.train_controlnet:
                        # Check if the index is valid
                        if i < len(controlnet):
                            controlnet_lora_dir = os.path.join(resume_dir, f"controlnet_lora_{i}")
                            adapter_model_path = os.path.join(controlnet_lora_dir, "adapter_model.safetensors")
                            if os.path.exists(adapter_model_path):
                                logger.info(f"Loading ControlNet LoRA for model {i} from {adapter_model_path}")
                                lora_state_dict = load_file(adapter_model_path, device="cpu")
                                controlnet[i].load_state_dict(lora_state_dict, strict=False)
                            else:
                                raise ValueError(f"Could not find ControlNet LoRA weights for model {i} at {adapter_model_path}. Please check the path.")
                        else:
                            logger.warning(f"Invalid index {i} for ControlNet model list of length {len(controlnet)}.")
                except Exception as e:
                    raise ValueError(f"Failed to load ControlNet LoRA weights: {e}")

            model_info_path = os.path.join(resume_dir, "model_info.json")
            if os.path.exists(model_info_path):
                with open(model_info_path, "r") as f:
                    model_info = json.load(f)
                    first_epoch = model_info.get("epoch", 0) + 1
                    global_step = model_info.get("global_step", 0)
                    best_score = model_info.get("best_score")
                    logger.info(f"Resuming with best score {best_score}")

            for model in [unet, text_encoder, vae, optimizer]:
                if model is not None and hasattr(model, "to"):
                    model.to(accelerator.device)
            free_memory()

            global_step = int(path.split("-")[1])
            initial_global_step = global_step
    else:
        initial_global_step = 0

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    resume_step = 0
    if args.resume_from_checkpoint and initial_global_step > 0:
        first_epoch = initial_global_step // num_update_steps_per_epoch
        resume_step = initial_global_step % num_update_steps_per_epoch
        logger.info(f"Resuming from epoch {first_epoch} and step {resume_step}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        if args.train_unet:
            unet.train()
        if args.train_controlnet:
            for i in args.train_controlnet:
                if i < len(controlnet):
                    controlnet[i].train()
        epoch_step_losses = []
        step_losses_since_update = []

        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    print("ERROR: NaN or Inf detected in latents after VAE encoding!")
                    break


                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                if random.random() < args.conditioning_dropout_prob:
                    encoder_hidden_states = text_encoder(torch.zeros_like(batch["input_ids"]))[0]
                else:
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # ControlNet: build the conditioning image(s) from one or two inputs.
                cond1 = batch["conditional_pixel_values_1"].to(dtype=weight_dtype)
                cond2 = batch["conditional_pixel_values_2"].to(dtype=weight_dtype)
                cond1_rgb = cond1 if cond1.shape[1] == 3 else cond1.repeat(1, 3, 1, 1)
                cond2_img = cond2 if cond2.shape[1] == 3 else cond2.repeat(1, 3, 1, 1)
                cond_image_1 = cond1_rgb
                cond_image_2 = cond2_img

                down_samples_list = []
                mid_samples_list = []
                cond_imgs = [cond_image_1, cond_image_2]
                for tcn, cimg in zip(controlnet, cond_imgs):
                    down_res, mid_res = tcn(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=cimg,
                        return_dict=False,
                    )
                    down_samples_list.append(down_res)
                    mid_samples_list.append(mid_res)

                if len(controlnet) > 1:
                    # Aggregate the residuals from all ControlNets
                    down_block_res_samples = [
                        sum(residuals) for residuals in zip(*down_samples_list)
                    ]
                    mid_block_res_sample = sum(mid_samples_list)
                elif len(controlnet) == 1:
                    # If there's only one ControlNet, no aggregation is needed
                    down_block_res_samples = down_samples_list[0]
                    mid_block_res_sample = mid_samples_list[0]

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / (snr + 1e-8)
                    )

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()
                if not torch.isfinite(loss):
                    tqdm.write(f"ERROR: Loss is {loss.item()}! Stopping training.")
                    break

                epoch_step_losses.append(loss.detach().item())
                step_losses_since_update.append(loss.detach().item())

                # Gather the losses across all processes for logging (if we use distributed training).
                if accelerator.use_distributed:
                    distributed_avg_loss = accelerator.gather(loss).mean()
                    train_loss += distributed_avg_loss.item() / args.gradient_accumulation_steps
                else:
                    train_loss += loss.detach().item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # Clip gradients for all trainable parameters
                    params_to_clip = all_trainable_params
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if accelerator.is_main_process:
                    optimizer_step_loss = np.mean(step_losses_since_update)
                    step_losses_since_update = []

                    current_lr = lr_scheduler.get_last_lr()[0]
                    accelerator.log({"loss": optimizer_step_loss, "lr": current_lr}, step=global_step)

                    avg_epoch_loss_to_log = None
                    if step == len(train_dataloader) - 1:
                        if epoch_step_losses:
                            avg_epoch_loss_to_log = np.nanmean(epoch_step_losses)
                            if not np.isnan(avg_epoch_loss_to_log):
                                tqdm.write(f"Epoch {epoch} average loss: {avg_epoch_loss_to_log:.4f}")

                    training_logger.log_training_step(
                        epoch=epoch,
                        global_step=global_step,
                        step_loss=optimizer_step_loss,
                        learning_rate=current_lr,
                        avg_epoch_loss=avg_epoch_loss_to_log
                    )

                    train_loss = 0.0


                if (accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED) and \
                    args.checkpointing_steps and global_step > 0 and global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            tqdm.write(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            tqdm.write(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)

                    try:
                        torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.bin"))
                    except Exception as e:
                        logger.warning(f"Failed to explicitly save optimizer.bin: {e}")
                    if args.train_unet:
                        unet_lora_save(accelerator, unet, save_path)

                    # Save ControlNet LoRA weights to a subdirectory
                    if args.train_controlnet:
                        controlnet_lora_save(args, accelerator, controlnet, save_path)
                    model_info = {
                        "epoch": epoch,
                        "best_score": best_score,
                        "global_step": global_step,
                    }
                    with open(os.path.join(save_path, "model_info.json"), "w") as f:
                        json.dump(model_info, f, indent=4)

                    logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)




        if accelerator.is_main_process and args.use_validation_split and val_dataloader is not None and epoch > 0 and epoch % args.validation_epochs == 0:
            logger.info("Running validation on validation dataset...")

            # Create pipeline for validation
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
                cache_dir=args.cache_dir,
                controlnet=accelerator.unwrap_model(controlnet),
            )
            
            # Load LoRA weights
            unwrapped_unet = accelerator.unwrap_model(unet)
            unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
            pipeline.load_lora_weights(unet_lora_state_dict)

            # Load ControlNet LoRA weights
            unwrapped_controlnet = accelerator.unwrap_model(controlnet)
            controlnet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_controlnet))
            pipeline.controlnet.load_state_dict(controlnet_lora_state_dict)

            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

            val_images = []
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None

            try:
                for i, batch in enumerate(val_dataloader):
                    if i >= args.num_validation_images:
                        break
                    
                    with torch.no_grad():
                        # Get the text embedding for conditioning
                        prompt_embeds = text_encoder(batch["input_ids"].to(accelerator.device))[0]
                        # Generate image
                        # Build condition image for evaluation

                        cond1 = batch["conditional_pixel_values_1"].to(accelerator.device, dtype=weight_dtype)
                        cond2 = batch["conditional_pixel_values_2"].to(accelerator.device, dtype=weight_dtype)
                        depth_gray = cond1.mean(dim=1, keepdim=True)
                        mask_gray = cond2.mean(dim=1, keepdim=True)
                        cond_image = torch.cat([depth_gray, mask_gray, depth_gray], dim=1)
                        image = pipeline(
                            prompt_embeds=prompt_embeds,
                            image=cond_image,
                            num_inference_steps=30,
                            generator=generator,
                        ).images[0]
                        

                    val_images.append(np.array(image))

            except Exception as e:
                logger.warning(f"Validation failed during image generation: {e}")

            if val_images:
                try:
                    tracker = accelerator.get_tracker("tensorboard")
                    if tracker:
                        tracker.add_images("validation", np.stack(val_images), epoch, dataformats="NHWC")
                        logger.info(f"Logged {len(val_images)} validation images to TensorBoard.")
                except Exception as e:
                    logger.warning(f"Failed to log validation images to TensorBoard: {e}")

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Optionally run comprehensive evaluation and save the best model
        if accelerator.is_main_process and getattr(args, "evaluation_epochs", 0) and epoch > 0 and (epoch % args.evaluation_epochs == 0 or epoch == args.num_train_epochs - 1) and hasattr(args, "_val_df") and not args._val_df.empty:

            # Set current epoch for evaluation
            args.current_epoch = epoch
            
            metrics = run_evaluation(
                args, 
                accelerator, 
                tokenizer, 
                noise_scheduler, 
                vae, 
                text_encoder, 
                unet, 
                controlnet,
                None, 
                weight_dtype,
                save_images=True, 
                dataset_type='val',
                conditional_image_dir_1=args.conditional_image_dir_1,
                conditional_image_dir_2=args.conditional_image_dir_2,
                controlnet_conditioning_scale=args.controlnet_conditioning_scale
            )

            current_score = metrics.get(args.best_model_metric, None)
            is_better = False
            if current_score is not None:
                is_better = current_score < best_score if args.lower_is_better else current_score > best_score

            if is_better:
                best_score = current_score
                logger.info(f"New best score for {args.best_model_metric}: {best_score:.4f}")
                best_model_dir = os.path.join(args.output_dir, "best_model")
                save_models(args, accelerator, unet, controlnet, best_model_dir, epoch, metrics) # room_counts_encoder, 

            # Log evaluation metrics
            training_logger.log_evaluation(epoch=epoch, metrics=metrics, is_best_model=is_better)

        if global_step >= args.max_train_steps:
            logger.info(f"Reached max_train_steps ({args.max_train_steps}). Finalizing logs and exiting training loop.")
            if accelerator.is_main_process:
                training_logger.finalize()
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Final evaluation
        if hasattr(args, "_val_df") and not args._val_df.empty:
            logger.info("Running final evaluation...")
            metrics = run_evaluation(
            args, 
            accelerator, 
            tokenizer, 
            noise_scheduler, 
            vae, 
            text_encoder, 
            unet, 
            controlnet,
            None, # Pass None to use the in-memory LoRA for final evaluation
            weight_dtype,
            save_images=True, # Always save images on final evaluation
            dataset_type='val',
            conditional_image_dir_1=args.conditional_image_dir_1,
            conditional_image_dir_2=args.conditional_image_dir_2,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale
            )

        # Always save the final model.
        logger.info("Saving last model.")
        last_model_dir = os.path.join(args.output_dir, "last_model")
        save_models(args, accelerator, unet, controlnet, last_model_dir, epoch, metrics) #room_counts_encoder, 

        # If no best model was saved, use the last model as the best one.
        best_model_path = os.path.join(args.output_dir, "best_model")
        if not os.path.exists(best_model_path):
            logger.info(
                "No best model was saved during training. Using last model as the best one."
            )
            # In case the best_model_path is a dangling symlink
            if os.path.islink(best_model_path):
                os.unlink(best_model_path)
            os.rename(last_model_dir, best_model_path)

    # Finalize logging (plots and summary)
    if accelerator.is_main_process:
        if args.eval_mode:
            training_logger.log_eval_results()
        training_logger.finalize()

if __name__ == "__main__":
    #warnings.simplefilter('error', UserWarning)
    main()
