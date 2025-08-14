import yaml
import sys
import os
import shutil
import argparse
from huggingface_hub import snapshot_download, hf_hub_download

def load_config(config_path):
    """Load the YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        raise

def download_checkpoints(repo_id, local_dir, allow_patterns=None, ignore_patterns=None):
    os.makedirs(local_dir, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type="model",
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            max_retries=3,  # Add retries
            timeout=600.0   # Set timeout to 10 minutes
        )
        print(f"Successfully downloaded checkpoints from {repo_id} to {local_dir}")
    except Exception as e:
        print(f"Error downloading checkpoints from {repo_id}: {e}")
        raise

def move_and_clean_checkpoints(local_dir, move_back_level=0):
    """Move files from checkpoints folder to local_dir and remove the empty folder."""
    source_dir = os.path.join(local_dir, "checkpoints")
    dest_dir = local_dir

    if move_back_level > 0:
        # Move up the directory tree by move_back_level
        for _ in range(move_back_level):
            dest_dir = os.path.dirname(dest_dir)

    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist. Check if checkpoints were downloaded.")
        return

    try:
        # Move all files and folders from source_dir to dest_dir
        for item in os.listdir(source_dir):
            source_path = os.path.join(source_dir, item)
            dest_path = os.path.join(dest_dir, item)
            if os.path.exists(dest_path):
                print(f"Skipping {item}: already exists in {dest_dir}")
                continue
            shutil.move(source_path, dest_path)
            print(f"Moved {item} to {dest_dir}")
        
        # Remove the empty checkpoints directory
        os.rmdir(source_dir)
        print(f"Removed empty directory: {source_dir}")
    except Exception as e:
        print(f"Error moving files or removing directory: {e}")
        raise

def copy_specific_files(source_dir, dest_dir="./ckpts"):
    """Copy specific files to the destination directory."""
    os.makedirs(dest_dir, exist_ok=True)
    files_to_copy = [
        "clap_music_speech_audioset_epoch_15_esc_89.98.pt",
        "hifigan_16k_64bins.json",
        "hifigan_16k_64bins.ckpt"
    ]

    try:
        for file in files_to_copy:
            source_path = os.path.join(source_dir, file)
            dest_path = os.path.join(dest_dir, file)
            if not os.path.exists(source_path):
                print(f"Source file {source_path} does not exist. Skipping.")
                continue
            if os.path.exists(dest_path):
                print(f"Skipping {file}: already exists in {dest_dir}")
                continue
            shutil.copy(source_path, dest_path)
            print(f"Copied {file} to {dest_dir}")
    except Exception as e:
        print(f"Error copying files: {e}")
        raise

def download_specific_checkpoint(repo_id, local_dir, checkpoint_path):
    """Download a specific checkpoint file from Hugging Face Hub."""
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=checkpoint_path,
            local_dir=local_dir,
            repo_type="model"
        )
        print(f"Successfully downloaded checkpoint {checkpoint_path} from {repo_id} to {local_dir}")
    except Exception as e:
        print(f"Error downloading checkpoint {checkpoint_path}: {e}")
        raise

def model_checkpoint_process(config, base_model_only=False, finetune_only=False):
    """Process checkpoints for models in the config based on base_model status."""
    if base_model_only and finetune_only:
        print("Error: Cannot select both --base_model_only and --finetune_only.")
        sys.exit(1)

    for model in config:
        repo_id = model.get("model_id")
        local_dir = model.get("local_dir")
        allow_patterns = model.get("allow")
        ignore_patterns = model.get("deny")
        platform = model.get("platform")
        is_base_model = model.get("base_model", False)
        move_back_level = model.get("move_back_level", 0)

        # Filter models based on base_model status
        if base_model_only and not is_base_model:
            print(f"Skipping {repo_id}: Not a base model")
            continue
        if finetune_only and is_base_model:
            print(f"Skipping {repo_id}: Not a finetuned model")
            continue

        if platform != "HuggingFace":
            print(f"Unsupported platform {platform} for model {repo_id}. Skipping.")
            continue

        # Download specific checkpoint if allow specifies a single file
        if allow_patterns and len(allow_patterns) == 1 and not allow_patterns[0].endswith("/*"):
            download_specific_checkpoint(repo_id, local_dir, allow_patterns[0])
        else:
            # Download folder/files with allow/deny patterns
            download_checkpoints(repo_id, local_dir, allow_patterns, ignore_patterns)
            move_and_clean_checkpoints(local_dir, move_back_level)
        
        copy_specific_files(local_dir, local_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download checkpoints from Hugging Face Hub based on config.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/model_ckpts.yaml",
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--base_model_only",
        action="store_true",
        help="Download only checkpoints for models marked as base_model: true (e.g., ayousanz/AudioLDM-training-finetuning, facebook/musicgen-small)"
    )
    parser.add_argument(
        "--finetune_only",
        action="store_true",
        help="Download only checkpoints for models marked as base_model: false (e.g., danhtran2mind/MusicGen-Small-MusicCaps-finetuning, danhtran2mind/AudioLDM-finetuning)"
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Download checkpoints for both base and finetuned models (default)"
    )
    args = parser.parse_args()

    # Validate arguments
    if sum([args.base_model_only, args.finetune_only, args.both]) > 1:
        print("Error: Only one of --base_model_only, --finetune_only, or --both can be specified.")
        sys.exit(1)

    # Default to both if no specific filter is provided
    if not args.base_model_only and not args.finetune_only and not args.both:
        args.both = True

    # Load and process the configuration
    config = load_config(args.config_path)
    model_checkpoint_process(config, base_model_only=args.base_model_only, finetune_only=args.finetune_only)