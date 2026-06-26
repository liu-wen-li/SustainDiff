"""
Training utilities for LoRA fine-tuning with conditional encoders.
"""
import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchvision import transforms
from diffusers import StableDiffusionControlNetPipeline, DiffusionPipeline, StableDiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers, is_wandb_available
from peft.utils import get_peft_model_state_dict
from evaluation import FlatLayoutEvaluator, FlatLayoutValidationDataset, split_dataset,get_train_transforms
from pathlib import Path
from diffusers.models.controlnets.controlnet import ControlNetModel
from diffusers.loaders import PeftAdapterMixin
from peft import PeftModel, PeftConfig, get_peft_model_state_dict,get_peft_model,set_peft_model_state_dict
from datetime import datetime
from typing import Optional, Union
import copy
from safetensors.torch import load_file
logger = logging.getLogger(__name__)
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None
    logging.warning("PEFT library not found. LoRA models in PEFT format will not be handled.")

logger = logging.getLogger(__name__)


def verify_lora_train(model_component):
    """
    Verifies if LoRA layers have been added to the UNet/ControlNet model(s) by checking for 'lora' in module class names.
    Supports single model, list/tuple of models, or a multi-controlnet wrapper.
    """
    def _has_lora(mod: torch.nn.Module) -> bool:
        try:
            return any('lora' in str(m.__class__).lower() for m in mod.modules())
        except Exception:
            return False

    if isinstance(model_component, (list, tuple)):
        # Require LoRA on all provided controlnets if training controlnet
        return all(_has_lora(m) for m in model_component)

    # Handle MultiControlNetModel style containers
    if hasattr(model_component, 'controlnets') and isinstance(getattr(model_component, 'controlnets'), (list, tuple)):
        return all(_has_lora(m) for m in getattr(model_component, 'controlnets'))

    return _has_lora(model_component)


def load_lora_into_controlnet(lora_state_dict, controlnet):
    """Merge LoRA weights directly into a ControlNet model in-place.

    For each `lora_A`/`lora_B` pair in `lora_state_dict`, the low-rank update
    ``delta = B @ A`` is added to the matching base weight of `controlnet`.

    Args:
        lora_state_dict (dict): LoRA weight dictionary keyed by parameter name.
        controlnet (ControlNetModel): Target ControlNet model instance.

    Returns:
        int: Number of weights that were successfully updated.
    """

    lora_pairs = {}
    for key in lora_state_dict.keys():
        if ".lora_A." in key:
            base_key = key.replace(".lora_A.weight", "")
            lora_pairs.setdefault(base_key, {})["A"] = lora_state_dict[key]
        elif ".lora_B." in key:
            base_key = key.replace(".lora_B.weight", "")
            lora_pairs.setdefault(base_key, {})["B"] = lora_state_dict[key]

    lora_check = 0
    for base_key, mats in lora_pairs.items():
        if base_key + ".weight" in controlnet.state_dict():
            base_weight = controlnet.state_dict()[base_key + ".weight"]
            A = mats.get("A")
            B = mats.get("B")
            if A is not None and B is not None:
                A = A.to(base_weight.device).to(base_weight.dtype)
                B = B.to(base_weight.device).to(base_weight.dtype)
                delta = 1 * (B @ A)
                base_weight.data += delta
                lora_check += 1
            else:
                print(f"Warning: incomplete lora matrices for {base_key}")
        else:
            print(f"Warning: {base_key}.base_layer.weight not found in model")
    return lora_check



def load_controlnet_lora_update(args, controlnet, resume_dir):
    if not isinstance(args.train_controlnet, list):
        args.train_controlnet = [args.train_controlnet]
    lora_check = 0
    for i in args.train_controlnet:
        if i < len(controlnet):
            controlnet_lora_dir = os.path.join(resume_dir, f"controlnet_lora_{i}")
            adapter_model_path = os.path.join(controlnet_lora_dir, "adapter_model.safetensors")
            if os.path.exists(adapter_model_path):
                logger.info(f"Loading ControlNet LoRA for model {i} from {adapter_model_path}")
                lora_state_dict = load_file(adapter_model_path, device="cpu")
                controlnet[i].load_state_dict(lora_state_dict, strict=False)
                lora_check+=1
    if lora_check == len(args.train_controlnet):
        return True
    else:
        return False


def build_pipeline(
    args,
    vae: torch.nn.Module,
    text_encoder: torch.nn.Module,
    tokenizer,
    unet: torch.nn.Module,
    controlnet: Union[ControlNetModel, object, list],
    noise_scheduler,
    lora_model_path: Optional[str] = None,
    device: str = "cuda"
):
    """
    A robust function to create a StableDiffusionControlNetPipeline, handling two LoRA loading modes.
    Mode 1 (Dynamic Loading): Pass base unet and controlnet, plus lora_model_path.
                      The function dynamically loads 'pytorch_lora_weights.safetensors' (UNet) and
                      merges the 'controlnet_lora' directory (ControlNet).
    Mode 2 (Pre-wrapped/Merged): Pass a unet and/or controlnet that are already PEFT-wrapped.
                         The function will automatically merge/unpack them into base models.
    Returns:
        - pipeline (StableDiffusionControlNetPipeline): The constructed pipeline object.
        - unet_lora_applied (bool): Flag indicating if UNet has LoRA applied.
        - controlnet_lora_applied (bool): Flag indicating if ControlNet has LoRA applied.
    """
    unet_lora_applied = False
    controlnet_lora_applied = False

    # --- Step 1: Pre-processing and LoRA Loading ---
    # Handle models that are already PEFT-wrapped from training (e.g., during mid-training validation)
    if not lora_model_path:
        if args.train_unet and verify_lora_train(unet):
            unet_lora_applied = True
        if args.train_controlnet:
            con_lora_check = 0
            for i in args.train_controlnet:
                if i < len(controlnet):
                    if verify_lora_train(controlnet[i]):
                        con_lora_check += 1
            if con_lora_check == len(args.train_controlnet):
                controlnet_lora_applied = True
    
    if lora_model_path and os.path.isdir(lora_model_path):
        if not isinstance(args.train_controlnet, list):
            args.train_controlnet = [args.train_controlnet]
        lora_check = 0
        for i in args.train_controlnet:
            lora_ck = 0
            if i < len(controlnet):
                controlnet_lora_dir = os.path.join(lora_model_path, f"controlnet_lora_{i}")
                adapter_model_path = os.path.join(controlnet_lora_dir, "adapter_model.safetensors")
                if os.path.exists(adapter_model_path):
                    try:
                        state_dict = load_file(adapter_model_path, device="cpu")
                        lora_ck = load_lora_into_controlnet(state_dict, controlnet[i])
                        if lora_ck > 0:
                            lora_check+=1
                    except Exception as e:
                        logger.error(f"Failed to load ControlNet LoRA for model {i}: {e}", exc_info=True)
        if lora_check == len(args.train_controlnet):
            controlnet_lora_applied = True
        else:
            controlnet_lora_applied = False
        
    # --- Step 2: Build Pipeline with processed components ---
    try:
        pipeline = StableDiffusionControlNetPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,  # unet is now LoRA-applied
            controlnet=controlnet,  # controlnet list is now LoRA-applied
            scheduler=noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
        )
        logger.info("Pipeline created successfully with LoRA-applied models.")
    except Exception as e:
        logger.error(f"Failed to create Pipeline: {e}", exc_info=True)
        raise

    # Dynamically load LoRA from a saved model path (for evaluation after training)
    # This modifies the unet and controlnet objects in-place before creating the pipeline.
    if lora_model_path and os.path.isdir(lora_model_path):
        # Load UNet LoRA
        if args.train_unet:
            unet_lora_file = "pytorch_lora_weights.safetensors"
            unet_lora_path = os.path.join(lora_model_path, unet_lora_file)
            if os.path.exists(unet_lora_path):
                try:
                    # This method loads LoRA layers into the UNet model directly
                    pipeline.load_lora_weights(lora_model_path, weight_name=unet_lora_file)
                    if any("lora" in k for k in pipeline.unet.state_dict().keys()):
                        unet_lora_applied = True
                        logger.info(f"Successfully loaded UNet LoRA from '{unet_lora_path}'. LoRA layers are present.")
                    else:
                        logger.warning(f"Loaded UNet LoRA from '{unet_lora_path}', but no new LoRA layers were found in the model.")
                except Exception as e:
                    raise RuntimeError(f"Failed to load UNet LoRA: {e}") from e




    pipeline.to(device)
    return pipeline, unet_lora_applied, controlnet_lora_applied


def flush_log_handlers():
    """Flushes all handlers on the root logger to ensure logs are written to disk."""
    root_logger = logging.getLogger()
    if hasattr(root_logger, "handlers"):
        for handler in root_logger.handlers:
            try:
                handler.flush()
            except Exception as e:
                # Log to stderr if flushing fails, as the logger itself might be failing.
                logger.warning(f"Warning: Failed to flush log handler {handler}: {e}")


def run_evaluation(args, accelerator, tokenizer, noise_scheduler, vae, text_encoder, unet, controlnet,
                lora_model_path, weight_dtype,save_images, dataset_type, conditional_image_dir_1, conditional_image_dir_2,controlnet_conditioning_scale): 
    """Run evaluation using the FlatLayoutEvaluator."""
    logger.info(f"Running evaluation for {dataset_type} dataset...")
    flush_log_handlers()

    # Create a deep copy of controlnet to avoid modifying the original model state during evaluation
    
    controlnet_for_eval = copy.deepcopy(controlnet)

    # Create validation dataset if not already available
    if dataset_type == 'val':
        if not hasattr(args, "_val_df") or args._val_df is None:
            raise ValueError("Validation dataframe not found in args._val_df. Please ensure dataset is split in main script.")
        eval_df = args._val_df
    elif dataset_type == 'test':
        if not hasattr(args, "_test_df") or args._test_df is None:
            raise ValueError("Test dataframe not found in args._test_df. Please ensure dataset is split in main script.")
        eval_df = args._test_df
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be 'val' or 'test'.")

    # Create validation dataset
    transform = get_train_transforms(args) 
    
    eval_image_paths = [os.path.join(args.train_data_dir, fname) for fname in eval_df['file_name']]
    eval_dataset = FlatLayoutValidationDataset(eval_image_paths, eval_df, transform=transform)
    # Track whether LoRA is applied to UNet and ControlNet to enforce requirement

    try:
        pipeline, unet_lora_applied, controlnet_lora_applied = build_pipeline(
            args=args,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet_for_eval,  # Use the copied controlnet for evaluation
            noise_scheduler=noise_scheduler,
            lora_model_path=lora_model_path,
            device=accelerator.device,
        )
        # ================
        if args.train_unet and not unet_lora_applied:
            raise ValueError("LoRA was not applied to the UNet.")
        if args.train_controlnet and not controlnet_lora_applied:
            raise ValueError("LoRA was not applied to the ControlNet.")

        logger.info("Pipeline built successfully.")
    except Exception as e:
        raise RuntimeError(f"Fatal error while building or checking the pipeline: {e}") from e

    # Store original training states
    is_unet_training = pipeline.unet.training
    is_controlnet_training = pipeline.controlnet.training

    # Set to eval mode for generation
    pipeline.unet.eval()
    pipeline.controlnet.eval()



    # Run evaluation
    evaluator = FlatLayoutEvaluator(device=accelerator.device.type)
    metrics = evaluator.evaluate_generation(
        args,
        pipeline,
        eval_dataset,
        args.eval_output_dir,
        conditional_image_dir_1=conditional_image_dir_1,
        conditional_image_dir_2=conditional_image_dir_2,
        num_samples=args.eval_num_samples,
        save_images=save_images,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        seed=args.seed,
    )
    # Restore original training states
    if is_unet_training:
        pipeline.unet.train()
    if is_controlnet_training:
        pipeline.controlnet.train()

    logger.info("Evaluation completed. Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    # Also persist raw metrics json for this epoch if available via args.current_epoch in caller
    try:
        epoch = args.current_epoch
    except Exception:
        epoch = None
    if accelerator.is_main_process:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = out_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        if epoch is not None:
            with open(metrics_dir / f"epoch_{epoch:04d}.json", "w", encoding="utf-8") as f:
                import json
                # Convert all values in metrics to standard Python types
                metrics = {key: (float(value) if isinstance(value, (torch.Tensor, np.floating)) else value) for key, value in metrics.items()}
                json.dump(metrics, f, ensure_ascii=False, indent=2)
        else:
            with open(metrics_dir / "latest.json", "w", encoding="utf-8") as f:
                import json
                # Convert all values in metrics to standard Python types
                metrics = {key: (float(value) if isinstance(value, (torch.Tensor, np.floating)) else value) for key, value in metrics.items()}
                json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def add_evaluation_args(parser):
    """Add evaluation-related arguments to the parser."""
    parser.add_argument(
        "--save_best_model", 
        action="store_true",
        help="Whether to save the best model based on validation metrics."
    )
    parser.add_argument(
        "--best_model_metric",
        type=str,
        default="fid",
        choices=["fid", "lpips"],
        help="Metric to use for determining the best model."
    )
    parser.add_argument(
        "--evaluation_epochs",
        type=int,
        default=0,
        help="Run comprehensive evaluation every X epochs (0 to disable)."
    )
    parser.add_argument(
        "--conditional_image_dir_1",
        type=str,
        default=None,
        help="Path to the directory containing conditional images."
    )
    parser.add_argument(
        "--conditional_image_dir_2",
        type=str,
        default=None,
        help="Path to the directory containing conditional images."
    )
    return parser