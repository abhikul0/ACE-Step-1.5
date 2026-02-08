#!/usr/bin/env python
"""Utility script to scan a directory of audio files, auto‑label all samples using ACE‑Step, and save the resulting dataset.

Usage:
    python scripts/label_all_dataset.py --input-dir /path/to/audio --output-dir /path/to/output

The script performs the following steps:
1. Initializes the DiT handler (AceStepHandler) and the 5Hz LM handler (LLMHandler).
2. Creates a DatasetBuilder instance.
3. Scans the input directory for audio files and optional .txt lyric files.
4. Auto‑labels every sample (caption, genre, BPM, key, etc.) using the DiT and LM models.
5. Saves the dataset as a JSON file in the specified output directory.
"""

import os
import argparse
import sys
from pathlib import Path

# Import core classes
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.training.dataset_builder import DatasetBuilder

import torch
torch.mps.set_per_process_memory_fraction(0.8)
print("BEFORE")
print(f"Allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
print(f"Driver total: {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")

def main():
    parser = argparse.ArgumentParser(description="Auto‑label all audio samples in a directory.")
    parser.add_argument("--input-dir", required=True, help="Path to directory containing audio files.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the resulting dataset JSON.")
    parser.add_argument("--config-path", default=os.getenv("ACESTEP_CONFIG_PATH", "acestep-v15-turbo"),
                        help="DiT model directory name (default: ACESTEP_CONFIG_PATH env or 'acestep-v15-turbo').")
    parser.add_argument("--lm-model-path", default=os.getenv("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-1.7B"),
                        help="5Hz LM model directory name (default: ACESTEP_LM_MODEL_PATH env or 'acestep-5Hz-lm-1.7B').")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "xpu"],
                        help="Device to run the DiT model on (default: auto).")
    parser.add_argument("--customTag", default="customTag", help="Path to directory containing audio files.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parent.parent

    # 1. Initialize DiT handler
    dit_handler = AceStepHandler()
    status_msg, _ = dit_handler.initialize_service(
        project_root="/Users/abhishek/acestep-test/ACE-Step-1.5",#str(project_root),
        config_path=args.config_path,
        device=args.device,
        use_flash_attention=False,
        compile_model=True,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
        quantization="int8_weight_only",
        prefer_source=None,
    )
    print(f"DiT init: {status_msg}")
    torch.mps.synchronize()
    torch.mps.empty_cache()

    # 2. Initialize LM handler
    lm_handler = LLMHandler()
    checkpoint_dir = project_root / "checkpoints"
    lm_status, lm_success = lm_handler.initialize(
        checkpoint_dir=str(checkpoint_dir),
        lm_model_path=args.lm_model_path,
        backend="pt",
        device="cpu",#args.device,
        offload_to_cpu=False,
        dtype=torch.bfloat16,
    )
    print(f"LM init: {lm_status}")
    if not lm_success:
        print("LM initialization failed. Exiting.")
        sys.exit(1)
    torch.mps.synchronize()
    torch.mps.empty_cache()

    # 3. Build dataset
    builder = DatasetBuilder()
    samples, scan_status = builder.scan_directory(str(input_dir))
    print(f"Scan status: {scan_status}")
    if not samples:
        print("No samples found. Exiting.")
        sys.exit(1)

    # 4. Apply tag and set instrumental
    builder.set_custom_tag(args.customTag)
    builder.set_all_instrumental(False)

    # 5. Auto‑label all samples
    def progress(msg):
        print(msg)

    result = builder.label_all_samples(
        dit_handler=dit_handler,
        llm_handler=lm_handler,
        format_lyrics=False,
        transcribe_lyrics=False,
        skip_metas=False,
        only_unlabeled=False,
        progress_callback=progress,
    )
    if isinstance(result, tuple) and len(result) == 3:
        table_data, label_status, _ = result
    else:
        table_data, label_status = result
    print(f"Labeling status: {label_status}")

    # 5. Save dataset
    dataset_name = input_dir.name
    save_path = output_dir / f"{dataset_name}.json"
    save_status = builder.save_dataset(str(save_path), dataset_name=dataset_name)
    print(f"Save status: {save_status}")
    print(f"Dataset saved to: {save_path}")


    print("AFTER")
    print(f"Allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
    print(f"Driver total: {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
