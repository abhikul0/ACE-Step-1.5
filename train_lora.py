import os
import argparse

from acestep.handler import AceStepHandler
from acestep.training.configs import LoRAConfig, TrainingConfig
from acestep.training.trainer import LoRATrainer

import torch
torch.mps.set_per_process_memory_fraction(0.6)
print("BEFORE")
print(f"Allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
print(f"Driver total: {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")

def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA adapters using pre‑processed tensors.")
    # Model and data paths
    parser.add_argument("--config_path", type=str, default="acestep-v15-turbo", help="DiT model configuration directory inside checkpoints.")
    parser.add_argument("--tensor_dir", type=str, default="preprocessed_tensors", help="Directory containing pre‑processed .pt tensors.")
    parser.add_argument("--output_dir", type=str, default="./lora_output", help="Directory to save checkpoints and final LoRA weights.")
    parser.add_argument("--export_path", type=str, default="./exported_lora", help="Directory to export final LoRA adapter.")
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate (overrides default in TrainingConfig).")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides default in TrainingConfig).")
    parser.add_argument("--gradient_accumulation", type=int, default=None, help="Gradient accumulation steps (overrides default).")
    parser.add_argument("--max_epochs", type=int, default=None, help="Maximum number of epochs (overrides default in TrainingConfig).")
    parser.add_argument("--save_every_n_epochs", type=int, default=None, help="Save checkpoint every N epochs (overrides default).")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Warmup steps for scheduler (overrides default).")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay (overrides default).")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="Max gradient norm (overrides default).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides default).")
    parser.add_argument("--num_workers", type=int, default=None, help="Data loader workers (overrides default).")
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory for data loader.")
    parser.add_argument("--log_every_n_steps", type=int, default=None, help="Log every N steps (overrides default).")
    # Device
    parser.add_argument("--device", type=str, default="auto", help="Device to run on (auto, cuda, mps, etc.).")
    parser.add_argument("--list-settings", action="store_true", help="Print all training settings and exit.")
    return parser.parse_args()


def print_all_settings():
    # Print a table of all settings with defaults
    settings = [
        ("config_path", "acestep-v15-turbo", "DiT model configuration directory inside checkpoints."),
        ("tensor_dir", "preprocessed_tensors", "Directory containing pre‑processed .pt tensors."),
        ("output_dir", "./lora_output", "Directory to save checkpoints and final LoRA weights."),
        ("export_path", "./exported_lora", "Directory to export final LoRA adapter."),
        ("learning_rate", None, "Learning rate (overrides default in TrainingConfig)."),
        ("batch_size", None, "Batch size (overrides default in TrainingConfig)."),
        ("gradient_accumulation", None, "Gradient accumulation steps (overrides default)."),
        ("max_epochs", None, "Maximum number of epochs (overrides default in TrainingConfig)."),
        ("save_every_n_epochs", None, "Save checkpoint every N epochs (overrides default)."),
        ("warmup_steps", None, "Warmup steps for scheduler (overrides default)."),
        ("weight_decay", None, "Weight decay (overrides default)."),
        ("max_grad_norm", None, "Max gradient norm (overrides default)."),
        ("seed", None, "Random seed (overrides default)."),
        ("num_workers", None, "Data loader workers (overrides default)."),
        ("pin_memory", False, "Pin memory for data loader."),
        ("log_every_n_steps", None, "Log every N steps (overrides default)."),
        ("device", "auto", "Device to run on (auto, cuda, mps, etc.)."),
    ]
    print("\nAvailable training settings:\n")
    for name, default, desc in settings:
        print(f"{name:25} {default!r:15} {desc}")


def main():
    args = parse_args()
    if args.list_settings:
        print_all_settings()
        return

    # Initialize handler
    handler = AceStepHandler()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    status_msg, _ = handler.initialize_service(project_root, args.config_path, device=args.device)
    print(f"Service init: {status_msg}")

    # Build configs
    lora_cfg = LoRAConfig()
    train_cfg = TrainingConfig(output_dir=args.output_dir)
    if args.learning_rate is not None:
        train_cfg.learning_rate = args.learning_rate
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.gradient_accumulation is not None:
        train_cfg.gradient_accumulation_steps = args.gradient_accumulation
    if args.max_epochs is not None:
        train_cfg.max_epochs = args.max_epochs
    if args.save_every_n_epochs is not None:
        train_cfg.save_every_n_epochs = args.save_every_n_epochs
    if args.warmup_steps is not None:
        train_cfg.warmup_steps = args.warmup_steps
    if args.weight_decay is not None:
        train_cfg.weight_decay = args.weight_decay
    if args.max_grad_norm is not None:
        train_cfg.max_grad_norm = args.max_grad_norm
    if args.seed is not None:
        train_cfg.seed = args.seed
    if args.num_workers is not None:
        train_cfg.num_workers = args.num_workers
    if args.pin_memory:
        train_cfg.pin_memory = True
    if args.log_every_n_steps is not None:
        train_cfg.log_every_n_steps = args.log_every_n_steps

    trainer = LoRATrainer(handler, lora_cfg, train_cfg)
    tensor_dir = os.path.abspath(os.path.join(project_root, args.tensor_dir))

    for step, loss, msg in trainer.train_from_preprocessed(tensor_dir):
        if loss is not None:
            print(f"[{step}] {msg} Loss: {loss:.4f}")
        else:
            print(f"[{step}] {msg}")

    print("AFTER")
    print(f"Allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
    print(f"Driver total: {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")

    # Export final LoRA adapter
    export_dir = os.path.abspath(os.path.join(project_root, args.export_path))
    os.makedirs(export_dir, exist_ok=True)
    # The trainer already saves final weights in train_cfg.output_dir/final
    # Copy them to export_dir
    import shutil
    src = os.path.join(train_cfg.output_dir, "final")
    if os.path.isdir(src):
        shutil.copytree(src, os.path.join(export_dir, "final"), dirs_exist_ok=True)
    print(f"Exported LoRA to {export_dir}")


if __name__ == "__main__":
    main()
