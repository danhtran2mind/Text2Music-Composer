import argparse
import yaml
import os
from huggingface_hub import snapshot_download

def load_config(config_path):
    """Load dataset configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def download_dataset(dataset_info):
    """Download a single dataset and save it to the specified local directory."""
    dataset_id = dataset_info['dataset_id']
    local_dir = dataset_info['local_dir']
    
    try:
        # Create local directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # Download dataset from HuggingFace using snapshot_download
        snapshot_download(repo_id=dataset_id, repo_type="dataset", local_dir=local_dir)
        
        print(f"Successfully downloaded {dataset_id} to {local_dir}")
    except Exception as e:
        print(f"Error downloading {dataset_id}: {str(e)}")

def main():
    """Main function to download datasets based on configuration."""
    parser = argparse.ArgumentParser(description="Download datasets from HuggingFace")
    parser.add_argument('--config_path', type=str, default="configs/datasets_info.yaml",
                        help='Path to the configuration YAML file')
    parser.add_argument('--dataset_id', type=str,
                        help='Optional Dataset ID to download (if not provided, downloads all datasets in config)')
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config_path)
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        return

    # Download datasets
    if args.dataset_id:
        # Download only the specified dataset
        for dataset_info in config:
            if dataset_info['dataset_id'] == args.dataset_id:
                download_dataset(dataset_info)
                return
        print(f"Dataset ID {args.dataset_id} not found in configuration")
    else:
        # Download all datasets in the config
        for dataset_info in config:
            download_dataset(dataset_info)

if __name__ == "__main__":
    main()