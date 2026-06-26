"""
Logging utilities for saving training and evaluation results.
"""
import os
import json
import csv
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
import torch


class TrainingLogger:
    """Logger for training losses and evaluation metrics."""
    
    def __init__(self, output_dir, experiment_name=None):
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create directories
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log files
        self.train_log_file = self.log_dir / f"{self.experiment_name}_training.csv"
        self.eval_log_file = self.log_dir / f"{self.experiment_name}_evaluation.csv"
        self.config_file = self.log_dir / f"{self.experiment_name}_config.json"
        
        # Initialize data storage
        self.training_logs = []
        self.evaluation_logs = []
        
        # Initialize CSV files with headers
        self._init_training_csv()
        self._init_evaluation_csv()
    
    def _init_training_csv(self):
        """Initialize training log CSV with headers."""
        if not self.train_log_file.exists():
            with open(self.train_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'global_step', 'step_loss', 'avg_epoch_loss', 
                    'learning_rate', 'timestamp'
                ])
    
    def _init_evaluation_csv(self):
        """Initialize evaluation log CSV with headers."""
        if not self.eval_log_file.exists():
            with open(self.eval_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'fid_score', 'lpips_score', 'clip_score',
                    'is_best_model', 'timestamp'
                ])
    
    def save_config(self, args):
        """Save training configuration."""
        config = {}
        if hasattr(args, '__dict__'):
            config = vars(args).copy()
        elif isinstance(args, dict):
            config = args.copy()
        
        # Convert non-serializable objects to strings
        for key, value in config.items():
            if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                config[key] = str(value)
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to {self.config_file}")
    
    def log_training_step(self, epoch, global_step, step_loss, learning_rate, avg_epoch_loss=None):
        """Log a training step."""
        timestamp = datetime.now().isoformat()
        
        # Store in memory
        log_entry = {
            'epoch': epoch,
            'global_step': global_step,
            'step_loss': step_loss,
            'avg_epoch_loss': avg_epoch_loss,
            'learning_rate': learning_rate,
            'timestamp': timestamp
        }
        self.training_logs.append(log_entry)
        
        # Append to CSV
        with open(self.train_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, global_step, step_loss, avg_epoch_loss, 
                learning_rate, timestamp
            ])
    
    def log_evaluation(self, epoch, metrics, is_best_model=False):
        """Log evaluation metrics."""
        timestamp = datetime.now().isoformat()
        
        # Store in memory
        log_entry = {
            'epoch': epoch,
            'metrics': metrics.copy(),
            'is_best_model': is_best_model,
            'timestamp': timestamp
        }
        self.evaluation_logs.append(log_entry)
        
        # Append to CSV
        with open(self.eval_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                metrics.get('fid', ''),
                metrics.get('lpips', ''),
                metrics.get('clip_score', ''),
                is_best_model,
                timestamp
            ])
    
    def plot_training_curves(self):
        """Generate and save training curve plots."""
        if not self.training_logs:
            print("No training logs to plot.")
            return
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(self.training_logs)
        
        # Plot training loss
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Step loss
        ax1.plot(df['global_step'], df['step_loss'], alpha=0.7, label='Step Loss')
        if 'avg_epoch_loss' in df.columns and df['avg_epoch_loss'].notna().any():
            # Plot average epoch loss
            epoch_df = df.dropna(subset=['avg_epoch_loss'])
            ax1.plot(epoch_df['global_step'], epoch_df['avg_epoch_loss'], 
                    color='red', linewidth=2, label='Epoch Avg Loss')
        
        ax1.set_xlabel('Global Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate
        ax2.plot(df['global_step'], df['learning_rate'], color='green')
        ax2.set_xlabel('Global Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plots_dir / f"{self.experiment_name}_training_curves.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        print(f"Training curves saved to {plot_path}")
    
    def plot_evaluation_metrics(self):
        """Generate and save evaluation metrics plots."""
        if not self.evaluation_logs:
            print("No evaluation logs to plot.")
            return
        
        # Extract metrics data
        epochs = []
        fid_scores = []
        lpips_scores = []
        clip_scores = []
        best_epochs = []

        for log in self.evaluation_logs:
            epochs.append(log['epoch'])
            fid_scores.append(log['metrics'].get('fid', None))
            lpips_scores.append(log['metrics'].get('lpips', None))
            clip_scores.append(log['metrics'].get('clip_score', None))
            if log['is_best_model']:
                best_epochs.append(log['epoch'])
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # FID Score (lower is better)
        if any(score is not None for score in fid_scores):
            valid_fid = [(e, s) for e, s in zip(epochs, fid_scores) if s is not None]
            if valid_fid:
                e_fid, s_fid = zip(*valid_fid)
                axes[0, 0].plot(e_fid, s_fid, 'b-o', linewidth=2, markersize=6)
                axes[0, 0].set_title('FID Score (Lower is Better)')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('FID Score')
                axes[0, 0].grid(True, alpha=0.3)
                # Mark best epochs
                for best_e in best_epochs:
                    if best_e in e_fid:
                        idx = e_fid.index(best_e)
                        axes[0, 0].scatter(best_e, s_fid[idx], color='red', s=100, zorder=5)
        
        # LPIPS Score (lower is better)
        if any(score is not None for score in lpips_scores):
            valid_lpips = [(e, s) for e, s in zip(epochs, lpips_scores) if s is not None]
            if valid_lpips:
                e_lpips, s_lpips = zip(*valid_lpips)
                axes[0, 1].plot(e_lpips, s_lpips, 'g-o', linewidth=2, markersize=6)
                axes[0, 1].set_title('LPIPS Score (Lower is Better)')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('LPIPS Score')
                axes[0, 1].grid(True, alpha=0.3)
                # Mark best epochs
                for best_e in best_epochs:
                    if best_e in e_lpips:
                        idx = e_lpips.index(best_e)
                        axes[0, 1].scatter(best_e, s_lpips[idx], color='red', s=100, zorder=5)
        
        # CLIP Score (higher is better)
        if any(score is not None for score in clip_scores):
            valid_clip = [(e, s) for e, s in zip(epochs, clip_scores) if s is not None]
            if valid_clip:
                e_clip, s_clip = zip(*valid_clip)
                axes[1, 0].plot(e_clip, s_clip, 'r-o', linewidth=2, markersize=6)
                axes[1, 0].set_title('CLIP Score (Higher is Better)')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('CLIP Score')
                axes[1, 0].grid(True, alpha=0.3)
                # Mark best epochs
                for best_e in best_epochs:
                    if best_e in e_clip:
                        idx = e_clip.index(best_e)
                        axes[1, 0].scatter(best_e, s_clip[idx], color='red', s=100, zorder=5)

        # Unused panel
        axes[1, 1].set_visible(False)

        plt.tight_layout()
        plot_path = self.plots_dir / f"{self.experiment_name}_evaluation_metrics.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        print(f"Evaluation metrics plot saved to {plot_path}")
    
    def save_summary(self):
        """Save a summary of the training session."""
        summary = {
            'experiment_name': self.experiment_name,
            'total_training_steps': len(self.training_logs),
            'total_evaluations': len(self.evaluation_logs),
            'final_timestamp': datetime.now().isoformat()
        }
        
        # Add training summary
        if self.training_logs:
            # Filter out logs with None step_loss
            valid_training_logs = [log for log in self.training_logs if log['step_loss'] is not None]
            if valid_training_logs:
                final_loss = valid_training_logs[-1]['step_loss']
                min_loss = min(log['step_loss'] for log in valid_training_logs)
                summary.update({
                    'final_training_loss': final_loss,
                    'minimum_training_loss': min_loss,
                    'final_learning_rate': valid_training_logs[-1]['learning_rate']
                })
        
        # Add evaluation summary
        if self.evaluation_logs:
            best_models = [log for log in self.evaluation_logs if log['is_best_model']]
            if best_models:
                best_model = best_models[-1]  # Last best model
                summary.update({
                    'best_model_epoch': best_model['epoch'],
                    'best_model_metrics': best_model['metrics']
                })
        
        summary_file = self.log_dir / f"{self.experiment_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Training summary saved to {summary_file}")
    
    def finalize(self):
        """Finalize logging by generating plots and summary."""
        print("Finalizing training logs...")
        self.plot_training_curves()
        self.plot_evaluation_metrics()
        self.save_summary()
        print("Logging finalized.")

def init_trackers_and_config(accelerator, args, project_name="sd3.5-lora"):
    """
    Initializes experiment trackers and saves the configuration.
    """
    if accelerator.is_main_process:
        def _to_basic(v):
            import numpy as _np
            from pathlib import Path as _Path
            if isinstance(v, (int, float, str, bool)) or v is None:
                return v
            try:
                import torch as _torch
                if isinstance(v, (_torch.dtype, _torch.device)):
                    return str(v)
            except Exception:
                pass
            if isinstance(v, (_np.generic,)):
                return v.item()
            if isinstance(v, (_Path, os.PathLike)):
                return str(v)
            if isinstance(v, (list, tuple)):
                try:
                    return ",".join(map(str, v))
                except Exception:
                    return str(v)
            if isinstance(v, dict):
                return str(v)
            return str(v)

        raw_cfg = dict(vars(args))
        safe_cfg = {k: _to_basic(v) for k, v in raw_cfg.items()}
        accelerator.init_trackers(project_name, config=safe_cfg)