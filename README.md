# SustainDiff: Automated End-to-End Generation of Residential Flat Layouts via a Sustainable Performance-Conditioned Diffusion Model

Code and dataset accompanying our paper
**"Automated End-to-End Generation of Residential Flat Layouts via a Sustainable
Performance-Conditioned Diffusion Model."**

**SustainDiff** is an energy-aware generative model for residential flat layouts. It
fine-tunes **Stable Diffusion v1.5** with **LoRA** adapters and **two ControlNet
branches**:

1. a **room-layout** condition (a simplified "bubble diagram" of room circles), and
2. an **energy-field** condition (a blurred field whose spread encodes the target
   energy-efficiency level),

so that generated flat layouts respect both the desired room layout and an energy
target.

---

## Repository structure

```
.
├── dataset/
│   ├── processed_dataset/          # Ground-truth flat layouts (the model's target output) — 54,648 PNGs
│   ├── conditional_images/         # ControlNet condition #1: room "bubble" circles — 54,648 PNGs
│   ├── energy_fields/              # ControlNet condition #2: blurred energy field — 54,648 PNGs
│   ├── dataset_splits/             # Fixed train/val/test partition used in the paper (file_name only)
│   │   ├── train_split.csv         # 53,548 rows
│   │   ├── val_split.csv           # 100 rows
│   │   └── test_split.csv          # 1,000 rows
│   ├── metadata.csv                # Per-image captions + attributes for all 54,648 images
│   └── energy_value.csv            # Per-image energy: raw value + normalized value
│
├── dataset_processing/             # Scripts that build the two condition inputs from flat layouts
│   ├── conditional_images.py       # flat layout -> room-circle condition
│   └── energy_mask.py              # flat layout -> blurred energy-field condition
│
├── train/                          # Training + evaluation pipeline
│   ├── main_controlnet.py          # Entry point (training and eval modes)
│   ├── training_utils.py           # Pipeline building, LoRA loading, evaluation driver
│   ├── evaluation.py               # Dataset splitting, transforms, FID/LPIPS evaluator
│   └── logging_utils.py            # CSV logging, loss/metric plots
│
├── evaluation/
│   └── results_evaluation.py       # Standalone metrics (MSE/MAE/PSNR/SSIM/LPIPS/FID/IoU)
│
├── requirements.txt
└── README.md
```

### `metadata.csv` columns

`metadata.csv` is the single source of per-image captions and attributes. The split
files reference rows here by `file_name`.

| Column | Description |
| --- | --- |
| `file_name` | Image file name, e.g. `0.png` (shared across all three image folders) |
| `text` | Text caption / prompt used for conditioning |
| `total_area` | Normalized total floor area |
| `room_type_0` … `room_type_11` | Per-room-type counts |

### Dataset splits

The three files in `dataset_splits/` define **only** the train/val/test partition —
each is a single `file_name` column. Captions and attributes are looked up from
`metadata.csv` by `file_name`. This keeps the partition separate from the metadata.

All three image folders use the **same file names**, so `0.png` in `processed_dataset/`,
`conditional_images/` and `energy_fields/` refer to the same flat layout.

> `energy_value.csv` holds the per-layout energy values, with columns `filename`,
> `energy_value` (raw, continuous) and `energy_normalization` (scaled to `[0, 1]`).
> The energy-field condition is built from `energy_normalization` (see
> "Regenerating the condition images"); the pre-built `energy_fields/` folder already
> encodes it, so this file is not needed for plain training.

---

## Downloading the dataset

The CSV files (`metadata.csv`, `dataset_splits/`, `energy_value.csv`) ship with this
repository. The three **image folders are hosted externally** because of their size,
on the Hugging Face Hub:

> **Download:** [liuwenli1207/sustaindiff-rplan-energy](https://huggingface.co/datasets/liuwenli1207/sustaindiff-rplan-energy)

The dataset is provided as three zip files — `processed_dataset.zip`,
`conditional_images.zip` and `energy_fields.zip`. You can grab them with the
`huggingface_hub` CLI:

```bash
pip install -U "huggingface_hub[cli]"
hf download liuwenli1207/sustaindiff-rplan-energy \
  --repo-type dataset --local-dir dataset
```

Then unzip each archive inside `dataset/` so the layout matches the tree above:

```bash
cd dataset
unzip -q processed_dataset.zip
unzip -q conditional_images.zip
unzip -q energy_fields.zip
```

Final layout:

```
dataset/
├── processed_dataset/      # extracted from processed_dataset.zip
├── conditional_images/     # extracted from conditional_images.zip
├── energy_fields/          # extracted from energy_fields.zip
├── dataset_splits/         # already in the repo
├── metadata.csv            # already in the repo
└── energy_value.csv        # already in the repo
```

---

## Installation

```bash
# 1. Create an environment (Python 3.10+ recommended)
conda create -n sustaindiff python=3.10 -y
conda activate sustaindiff

# 2. Install PyTorch matching your CUDA version (see https://pytorch.org)
#    Example (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install the remaining dependencies
pip install -r requirements.txt
```

Optional extras, only needed for specific flags:

- `bitsandbytes` — for `--use_8bit_adam`
- `xformers` — for `--enable_xformers_memory_efficient_attention`
- `wandb` — for `--report_to wandb`

---

## Base models

The pipeline downloads these from the Hugging Face Hub on first run (cached in
`--cache_dir`):

- UNet / VAE / text encoder: [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- ControlNet #1 (room layout): [`lllyasviel/control_v11p_sd15_seg`](https://huggingface.co/lllyasviel/control_v11p_sd15_seg)
- ControlNet #2 (energy field): [`lllyasviel/control_v11f1p_sd15_depth`](https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth)

---

## Training

Run from the `train/` directory. The command below trains LoRA on the UNet and on
ControlNet #0 (room layout), using both condition images:

```bash
cd train

accelerate launch main_controlnet.py \
  --pretrained_model_name_or_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --controlnet_model_name_or_path lllyasviel/control_v11p_sd15_seg lllyasviel/control_v11f1p_sd15_depth \
  --train_data_dir   ../dataset/processed_dataset \
  --metadata_path    ../dataset/metadata.csv \
  --dataset_path     ../dataset \
  --conditional_image_dir_1 ../dataset/conditional_images \
  --conditional_image_dir_2 ../dataset/energy_fields \
  --output_dir       ./runs/sustaindiff_lora \
  --eval_output_dir  ./runs/sustaindiff_lora_eval \
  --cache_dir        ./hf_cache \
  --resolution 512 \
  --train_batch_size 16 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 2 \
  --learning_rate 1e-4 \
  --lr_scheduler cosine --lr_warmup_steps 0 \
  --max_grad_norm 1 \
  --mixed_precision bf16 \
  --train_unet \
  --train_controlnet 0 \
  --unet_rank 32 \
  --controlnet_rank 128 \
  --use_validation_split \
  --val_set_size 100 \
  --test_set_size 1000 \
  --evaluation_epochs 1 \
  --eval_num_samples 100 \
  --save_best_model \
  --best_model_metric fid \
  --checkpointing_steps 800 \
  --checkpoints_total_limit 1 \
  --center_crop \
  --allow_tf32 \
  --use_8bit_adam \
  --report_to tensorboard \
  --seed 1337 \
  --controlnet_conditioning_scale_1 1.0 \
  --controlnet_conditioning_scale_2 1.0
```

To train **both** ControlNets, pass `--train_controlnet 0 1`.

> **Captions / metadata.** `--metadata_path ../dataset/metadata.csv` supplies the
> per-image caption (`text` column), keyed by `file_name`. If you omit it, every
> image falls back to the default caption
> (`"A flat layout of residential buildings. LWL_STYLE_FLAT_LAYOUT"`). The split CSVs
> hold the partition only; captions are always looked up from `metadata.csv`.

### Key arguments

| Argument | Meaning |
| --- | --- |
| `--pretrained_model_name_or_path` | Base SD model (Hub id or local path) |
| `--controlnet_model_name_or_path` | One or two ControlNet models (space-separated) |
| `--train_data_dir` | Folder of ground-truth flat layouts (`processed_dataset/`) |
| `--metadata_path` | Metadata CSV with `file_name` + `text` columns (`metadata.csv`); omit to use the default caption |
| `--dataset_path` | Dataset root; a `processed_dataset_cache/` is created here |
| `--conditional_image_dir_1/2` | The two ControlNet condition folders |
| `--output_dir` | Where checkpoints, splits, logs and best/last model are saved |
| `--eval_output_dir` | Where evaluation images and metrics are written |
| `--cache_dir` | Hugging Face download cache |
| `--train_unet` / `--train_controlnet` | Which components get LoRA adapters |
| `--unet_rank` / `--controlnet_rank` | LoRA rank for each component |
| `--use_validation_split` | Enable train/val/test splitting |
| `--val_set_size` / `--test_set_size` | Fixed split sizes (override ratios) |
| `--evaluation_epochs` | Run full FID/LPIPS evaluation every N epochs (0 = off) |
| `--save_best_model` / `--best_model_metric` | Keep the best checkpoint by a metric (`fid`/`lpips`) |

Run `python main_controlnet.py --help` for the full list.

### Reproducing the exact paper splits

When `--use_validation_split` is set, the script looks for
`train_split.csv`, `val_split.csv`, `test_split.csv` inside `<output_dir>/splits/`.
If they exist it **reuses** them; otherwise it creates new random splits and saves
them there. Each split file is a single `file_name` column; captions/attributes are
joined from `metadata.csv` at load time.

To reproduce the paper's exact splits, copy the provided CSVs into that folder
before training:

```bash
mkdir -p train/runs/sustaindiff_lora/splits
cp dataset/dataset_splits/*.csv train/runs/sustaindiff_lora/splits/
```

---

## Evaluation

### Option A — evaluate during/after training

Add `--eval_mode` to load the saved model from `--output_dir` (it tries
`best_model/`, then `last_model/`, then the latest `checkpoint-*`) and run FID/LPIPS
on the test set:

```bash
cd train

accelerate launch main_controlnet.py \
  --pretrained_model_name_or_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --controlnet_model_name_or_path lllyasviel/control_v11p_sd15_seg lllyasviel/control_v11f1p_sd15_depth \
  --train_data_dir   ../dataset/processed_dataset \
  --metadata_path    ../dataset/metadata.csv \
  --dataset_path     ../dataset \
  --conditional_image_dir_1 ../dataset/conditional_images \
  --conditional_image_dir_2 ../dataset/energy_fields \
  --output_dir       ./runs/sustaindiff_lora \
  --eval_output_dir  ./runs/sustaindiff_lora_eval \
  --cache_dir        ./hf_cache \
  --resolution 512 \
  --mixed_precision bf16 \
  --train_unet --train_controlnet 0 \
  --unet_rank 32 --controlnet_rank 128 \
  --use_validation_split --val_set_size 100 --test_set_size 1000 \
  --eval_mode --eval_num_samples 100 \
  --seed 1337
```

Generated and real images are written under `--eval_output_dir`, and metrics to
`evaluation_metrics.json`.

### Option B — standalone metrics on two folders

`evaluation/results_evaluation.py` compares any two folders of images (matched by
file name) and reports MSE, MAE, PSNR, SSIM, LPIPS, FID and per-class IoU:

```bash
python evaluation/results_evaluation.py \
  --real_dir      <path/to/real_images> \
  --generated_dir <path/to/generated_images>
```

Results are printed and saved to `image_evaluation_results_withstd.csv` next to the
generated folder.

---

## Regenerating the condition images (optional)

The dataset already ships with both condition folders. To rebuild them from a set of
flat-layout masks:

```bash
# 1. Room-circle condition
python dataset_processing/conditional_images.py \
  --input_dir  <path/to/flat_layouts> \
  --output_dir dataset/conditional_images

# 2. Energy-field condition (needs a CSV with filename + energy_normalization,
#    i.e. dataset/energy_value.csv)
python dataset_processing/energy_mask.py \
  --input_dir  dataset/conditional_images \
  --output_dir dataset/energy_fields \
  --energy_csv dataset/energy_value.csv
```

---

## Notes

- The large image folders (`processed_dataset/`, `conditional_images/`,
  `energy_fields/`) are **not** stored in git; download them separately (see
  "Downloading the dataset" above).
- Training writes a `main_log.txt` in the working directory and TensorBoard logs
  under `<output_dir>/logs`.

---

## License

This project is released under the **Apache License 2.0** (see [`LICENSE`](LICENSE)).
</content>
