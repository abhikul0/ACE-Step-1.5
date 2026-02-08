import argparse
import os
import sys

from pathlib import Path

import torch

from acestep.training.dataset_builder import DatasetBuilder
from acestep.handler import AceStepHandler


torch.mps.set_per_process_memory_fraction(0.6)
print("BEFORE")
print(f"Allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
print(f"Driver total: {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")

def main():
    parser = argparse.ArgumentParser(description="Preprocess a dataset JSON to tensors for ACE-Step training.")
    parser.add_argument("--dataset_path", help="Path to the dataset JSON file generated in Step 4.")
    parser.add_argument("--output_dir", help="Directory to store the preprocessed .pt files and manifest.json.")
    parser.add_argument(
        "--config_path",
        default="acestep-v15-turbo",
        help="Name of the DiT model configuration directory (default: 'acestep-v15-turbo').",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=240.0,
        help="Maximum duration (seconds) of audio to process per sample.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_file():
        print(f"❌ Dataset file not found: {dataset_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    builder = DatasetBuilder()
    samples, status = builder.load_dataset(str(dataset_path))
    print(status)

    if not samples:
        print("❌ No samples loaded. Exiting.")
        sys.exit(1)

    # Initialize handler (model service)
    handler = AceStepHandler()
    # Determine project root (current working directory)
    project_root = Path.cwd()
    config_path = args.config_path

    status_msg, _ = handler.initialize_service(
        project_root=str(project_root),
        config_path=config_path,
        device="auto",
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
        compile_model=True,
        quantization = "int8_weight_only"
    )
    print(status_msg)

    # Preprocess to tensors
    output_paths, preprocess_status = builder.preprocess_to_tensors(
        handler, str(output_dir), max_duration=args.max_duration
    )
    print(preprocess_status)

    if output_paths:
        print(f"✅ Preprocessed tensors saved to {output_dir}")
    else:
        print("❌ No tensors were created.")

    print("AFTER")
    print(f"Allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
    print(f"Driver total: {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
