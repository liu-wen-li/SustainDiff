import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from diffusers import DiffusionPipeline
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import json
from tqdm import tqdm
import logging
from accelerate.logging import MultiProcessAdapter
import csv

# Default caption used when no metadata CSV is provided (all images share one prompt).
DEFAULT_CAPTION = "A flat layout of residential buildings. LWL_STYLE_FLAT_LAYOUT"


def get_train_transforms(args):
    """Returns a composition of transformations for training."""
    return transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

def conditional_image_transforms(args):
    """Returns a composition of transformations for conditional images."""
    return transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

def cleanup_pipeline_memory(pipeline=None):
    """Free a pipeline and clear the CUDA cache."""
    if pipeline is not None:
        del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def cleanup_model_memory(model=None):
    """Free a model and clear the CUDA cache."""
    if model is not None:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# Import FID calculation
try:
    from torch_fidelity import calculate_metrics
    FID_AVAILABLE = True
except ImportError:
    print("Warning: torch-fidelity not installed. FID calculation will be disabled.")
    FID_AVAILABLE = False

# Import LPIPS for perceptual similarity
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("Warning: lpips not installed. LPIPS calculation will be disabled.")
    LPIPS_AVAILABLE = False



logger = logging.getLogger(__name__)

class FlatLayoutValidationDataset(Dataset):
    """Validation dataset for flat layout generation with conditional inputs."""
    
    def __init__(self, image_paths: List[str], metadata_df: pd.DataFrame, 
                 transform: Optional[transforms.Compose] = None):
        """
        Args:
            image_paths: List of image file paths
            metadata_df: DataFrame containing metadata with columns: 
                        file_name, text
            transform: Image transforms to apply
        """
        self.image_paths = image_paths
        self.metadata_df = metadata_df
        self.transform = transform
        
        # Create lookup for fast metadata access
        self.metadata_lookup = {}
        for _, row in metadata_df.iterrows():
            self.metadata_lookup[row['file_name']] = row
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        file_name = os.path.basename(image_path)
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Get metadata
        metadata = self.metadata_lookup[file_name]
        
        return {
            'image': image,
            'file_name': file_name,
            'text': metadata['text'],
        }

class FlatLayoutEvaluator:
    """
    Comprehensive evaluator for flat layout generation models.
    Supports FID and LPIPS metrics.
    """

    def __init__(self, device: str = 'cuda'):
        """
        Args:
            device: Device to run evaluation on
        """
        self.device = device

        # Initialize evaluation models
        self._init_evaluation_models()
    def cleanup(self):
        """Free the evaluator's models."""
        if hasattr(self, 'lpips_model') and self.lpips_model is not None:
            cleanup_model_memory(self.lpips_model)
            self.lpips_model = None
    
    def _init_evaluation_models(self):
        """Initialize models for different evaluation metrics."""
        # LPIPS model
        if LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net='alex', pretrained=True).to(self.device)
            self.lpips_model.eval()
        
    
    def calculate_fid(self, args,
                      real_images_path: str, 
                      generated_images_path: str) -> float:
        """
        Calculate FID score between real and generated images using torch-fidelity.
        
        Args:
            real_images_path: Path to directory containing real images
            generated_images_path: Path to directory containing generated images
            
        Returns:
            FID score (lower is better), or -1.0 on failure.
        """
        if not FID_AVAILABLE:
            logger.warning("FID calculation skipped - torch-fidelity not available")
            return -1.0
        
        logger.info(f"Calculating FID between {real_images_path} and {generated_images_path}")
        try:
            metrics = calculate_metrics(
                input1=generated_images_path,
                input2=real_images_path,
                cuda=(self.device == 'cuda'),
                isc=False,
                fid=True,
                kid=False,
                verbose=False,
                feature_layer_fid='2048',
                batch_size=args.eval_num_samples
            )
            fid_score = metrics['frechet_inception_distance']
            logger.info(f"FID score: {fid_score:.4f}")
            return fid_score
        except Exception as e:
            logger.error(f"Failed to calculate FID score: {e}", exc_info=True)
            return -1.0  # Indicate failure

    def calculate_lpips(self, 
                        real_images: List[Image.Image], 
                        generated_images: List[Image.Image]) -> float:
        """
        Calculate average LPIPS score between paired real and generated images.
        
        Args:
            real_images: List of real PIL Images
            generated_images: List of generated PIL Images
            
        Returns:
            Average LPIPS score (lower is better)
        """
        try:
            if not LPIPS_AVAILABLE:
                logger.warning("LPIPS calculation skipped - lpips not available")
                return float('inf')
            
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            
            lpips_scores = []
            for real_img, gen_img in zip(real_images, generated_images):
                real_tensor = transform(real_img).unsqueeze(0).to(self.device)
                gen_tensor = transform(gen_img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    score = self.lpips_model(real_tensor, gen_tensor)
                    lpips_scores.append(score.item())
            
            return np.mean(lpips_scores)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def evaluate_generation(self,
                          args,
                          pipeline,
                          validation_dataset: FlatLayoutValidationDataset,
                          output_dir: str,
                          conditional_image_dir_1: str,
                          conditional_image_dir_2: str,
                          num_samples: int = None,
                          num_inference_steps: int = 30,
                          guidance_scale: float = 7.5,
                          seed: int = None,
                          save_images: bool = True,
                          controlnet_conditioning_scale: Union[float, List[float]] = 1.0) -> Dict[str, float]:
        """
        Comprehensive evaluation of the generation model.
        
        Args:
            pipeline: Diffusion pipeline for generation
            validation_dataset: Validation dataset
            output_dir: Directory to save generated images
            conditional_image_dir_1: Directory containing first type of conditional images.
            conditional_image_dir_2: Directory containing second type of conditional images.
            num_samples: Number of samples to evaluate (None for all)
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale for generation
            seed: Random seed for reproducible generation
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        try:
            # Ensure pipeline components are in evaluation mode
            if hasattr(pipeline, 'unet'):
                pipeline.unet.eval()
            if hasattr(pipeline, 'vae'):
                pipeline.vae.eval()
            if hasattr(pipeline, 'text_encoder'):
                pipeline.text_encoder.eval()
            if hasattr(pipeline, 'controlnet'):
                pipeline.controlnet.eval()

            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'generated'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'generated_1'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'real'), exist_ok=True)
            evaluation_log_path = os.path.join(output_dir, "evaluation_log.txt")
            file_handler = logging.FileHandler(evaluation_log_path)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            target_logger = logger.logger if isinstance(logger, MultiProcessAdapter) else logger
            target_logger.addHandler(file_handler)
            
            # Setup data loader
            dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
            if num_samples:
                total_samples = min(num_samples, len(validation_dataset))
            else:
                total_samples = len(validation_dataset)
            
            # Generation setup
            generator = torch.Generator(device=pipeline.device)
            if seed is not None:
                generator.manual_seed(seed)
            
            # Storage for evaluation data
            generated_images = []
            real_images = []
            texts = []
            
            # Generate images
            logger.info(f"Generating {total_samples} images for evaluation...")
            pipeline.set_progress_bar_config(disable=True)
            
            for i, batch in enumerate(tqdm(dataloader)):
                if i >= total_samples:
                    break
                
                # Extract batch data
                text = batch['text'][0]
                # room_counts = batch['room_counts'][0]
                real_image = batch['image'][0]
                file_name = batch['file_name'][0]

                # Load and prepare conditional images
                conditional_image_path_1 = os.path.join(conditional_image_dir_1, file_name)
                conditional_image_path_2 = os.path.join(conditional_image_dir_2, file_name)
                try:
                    conditional_image_1 = Image.open(conditional_image_path_1).convert("RGB")
                    conditional_image_2 = Image.open(conditional_image_path_2).convert("RGB")

                    image_transform = conditional_image_transforms(args)
                    conditional_image_1 = image_transform(conditional_image_1).unsqueeze(0).to(device=pipeline.device, dtype=pipeline.dtype)
                    conditional_image_2 = image_transform(conditional_image_2).unsqueeze(0).to(device=pipeline.device, dtype=pipeline.dtype)
                except FileNotFoundError:
                    logger.error(f"Conditional images not found for {file_name}. Skipping sample.")
                    continue
                
                # Prepare scale for multiple controlnets
                scale = controlnet_conditioning_scale
                if not isinstance(scale, (list, tuple)):
                    scale = [scale, scale]
                
                # Generate image with conditional inputs
                with torch.autocast(device_type='cuda', dtype=torch.float16):

                    generated_image = generate_with_conditions(
                        pipeline=pipeline,
                        text=text,
                        image=[conditional_image_1, conditional_image_2],
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        controlnet_conditioning_scale=scale
                    )
                    generated_image_1 = pipeline(
                        text,
                        image=[conditional_image_1, conditional_image_2],
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        controlnet_conditioning_scale=scale,
                    ).images[0]
                
                # Convert real image tensor to PIL for evaluation and saving
                # The real_image tensor is normalized to [-1, 1], need to convert back to [0, 1]
                real_image_denorm = (real_image + 1.0) / 2.0
                real_pil = transforms.ToPILImage()(real_image_denorm)

                # Save images if required
                if save_images:
                    generated_image.save(os.path.join(output_dir, 'generated', file_name))
                    generated_image_1.save(os.path.join(output_dir, 'generated_1', file_name.replace('.png', '_1.png')))
                    real_pil.save(os.path.join(output_dir, 'real', file_name))
                
                # Store for evaluation
                generated_images.append(generated_image)
                real_images.append(real_pil)
                texts.append(text)
                # room_counts_list.append(room_counts)

            logger.info("Generation completed. Starting cleanup...")
            if 'generator' in locals():
                del generator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Calculate evaluation metrics
            logger.info("Calculating evaluation metrics...")
            metrics = {}
            
            # FID Score
            if FID_AVAILABLE:
                metrics['fid'] = self.calculate_fid(args,
                    real_images_path=os.path.join(output_dir, 'real'),
                    generated_images_path=os.path.join(output_dir, 'generated')
                )
            
            # LPIPS Score
            if LPIPS_AVAILABLE:
                lpips_score = self.calculate_lpips(real_images, generated_images)
                metrics['lpips'] = lpips_score

            # Save metrics
            with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
                # Convert all values in metrics to standard Python types
                metrics = {key: (float(value) if isinstance(value, (torch.Tensor, np.floating)) else value) for key, value in metrics.items()}
                json.dump(metrics, f, indent=2)

            logger.info("Evaluation completed. Metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")
        
            return metrics
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()


def split_dataset(image_dir: str,
                  metadata_path: str = None,
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  test_ratio: float = 0.1,
                  random_state: int = 42,
                  output_dir: str = None,
                  val_set_size: Optional[int] = None,
                  test_set_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/validation/test sets.
    The logic prioritizes fixed sizes (`val_set_size`, `test_set_size`) if provided.
    If they are not, it falls back to using ratios.

    If `metadata_path` is None, the file list is built from the images in
    `image_dir` and every row is given the default caption.
    """
    # Check for existing splits first if output_dir is provided
    if output_dir is not None:
        splits_dir = os.path.join(output_dir, "splits")
        train_split_path = os.path.join(splits_dir, "train_split.csv")
        val_split_path = os.path.join(splits_dir, "val_split.csv")
        test_split_path = os.path.join(splits_dir, "test_split.csv")

        if all(os.path.exists(p) for p in [train_split_path, val_split_path, test_split_path]):
            logger.info(f"Found existing dataset splits in {splits_dir}, loading them...")
            try:
                train_df = pd.read_csv(train_split_path)
                val_df = pd.read_csv(val_split_path)
                test_df = pd.read_csv(test_split_path)

                def verify_files_exist(df, split_name):
                    for fname in df['file_name']:
                        if not os.path.exists(os.path.join(image_dir, fname)):
                            raise FileNotFoundError(f"Image file '{fname}' from {split_name} split not found in '{image_dir}'")

                verify_files_exist(train_df, "train")
                verify_files_exist(val_df, "validation")
                verify_files_exist(test_df, "test")

                logger.info(f"Successfully loaded and verified existing splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
                return train_df, val_df, test_df
            except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
                logger.warning(f"Failed to load or verify existing splits: {e}. Will create new splits.")

    # Build the file list from a metadata CSV, or from the image folder directly.
    available_images = set(os.listdir(image_dir))
    if metadata_path:
        df = pd.read_csv(metadata_path)
        df = df[df['file_name'].isin(available_images)].copy()
    else:
        png_files = sorted(f for f in available_images if f.lower().endswith('.png'))
        df = pd.DataFrame({
            'file_name': png_files,
            'text': DEFAULT_CAPTION,
        })
    
    # Shuffle the dataset for random splitting
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    if val_set_size is not None and test_set_size is not None:
        logger.info(f"Using fixed sizes for splitting: Val={val_set_size}, Test={test_set_size}")
        
        if val_set_size + test_set_size >= len(df):
            raise ValueError("Sum of val_set_size and test_set_size must be smaller than the total dataset size.")

        test_df = df.iloc[:test_set_size]
        val_df = df.iloc[test_set_size : test_set_size + val_set_size]
        train_df = df.iloc[test_set_size + val_set_size:]
    else:
        logger.info("Using ratios for dataset splitting.")
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        train_df, temp_df = train_test_split(df, test_size=(val_ratio + test_ratio), random_state=random_state)
        
        if len(temp_df) > 0:
            relative_test_ratio = test_ratio / (val_ratio + test_ratio)
            val_df, test_df = train_test_split(temp_df, test_size=relative_test_ratio, random_state=random_state)
        else:
            val_df, test_df = pd.DataFrame(), pd.DataFrame()

    logger.info(f"Dataset split completed: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Save the new splits if output_dir is provided. Splits define the partition
    # only, so we persist just the `file_name` column; captions/attributes are
    # looked up separately from the metadata CSV.
    if output_dir is not None:
        splits_dir = os.path.join(output_dir, "splits")
        os.makedirs(splits_dir, exist_ok=True)
        train_df[["file_name"]].to_csv(os.path.join(splits_dir, "train_split.csv"), index=False)
        val_df[["file_name"]].to_csv(os.path.join(splits_dir, "val_split.csv"), index=False)
        test_df[["file_name"]].to_csv(os.path.join(splits_dir, "test_split.csv"), index=False)
        logger.info(f"Saved new dataset splits to {splits_dir}")
        
    return train_df, val_df, test_df

    
    

def generate_with_conditions(pipeline,
                           text: str,
                           image: Image.Image,
                           num_inference_steps: int,
                           guidance_scale: float,
                           generator: torch.Generator = None,
                           controlnet_conditioning_scale: float = 1.0) -> Image.Image:
    """
    Generate an image from a text prompt and ControlNet condition image(s),
    applying classifier-free guidance with explicit positive/negative embeddings.

    Returns:
        Generated PIL image.
    """
    device = pipeline.device
    do_classifier_free_guidance = guidance_scale > 1.0

    with torch.no_grad():
        # Encode the text prompt (positive and, for CFG, negative).
        prompt_embeds = pipeline._encode_prompt(
            prompt=text,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        if do_classifier_free_guidance:
            negative_prompt_embeds, prompt_embeds = torch.chunk(prompt_embeds, 2)
        else:
            negative_prompt_embeds = None

        try:
            result = pipeline(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
            )
            image = result.images[0]
        except Exception as e:
            print(f"Error during image generation: {e}")

        return image