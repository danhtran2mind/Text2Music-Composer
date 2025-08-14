import argparse
import yaml
import os
from huggingface_hub import snapshot_download

def load_config(config_path):
    """Load dataset configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def download_dataset(dataset_info, local_data_dir):
    """Download a single dataset and save it to the specified local directory."""
    dataset_name = dataset_info['dataset_name']
    local_dir = os.path.join(local_data_dir, dataset_info['local_dir'])
    
    try:
        # Create local directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # Download dataset from HuggingFace using snapshot_download
        snapshot_download(repo_id=dataset_name, repo_type="dataset", local_dir=local_dir)
        
        print(f"Successfully downloaded {dataset_name} to {local_dir}")
    except Exception as e:
        print(f"Error downloading {dataset_name}: {str(e)}")

def main():
    """Main function to download datasets based on configuration."""
    parser = argparse.ArgumentParser(description="Download datasets from HuggingFace")
    parser.add_argument('--local_dir', type=str, default='./data',
                        help='Local directory to save datasets')
    args = parser.parse_args()

    # Define path to configuration file
    config_path = os.path.join('configs', 'datasets_info.yaml')
    
    # Load configuration
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        return

    # Download each dataset
    for dataset_info in config:
        download_dataset(dataset_info, args.local_dir)

if __name__ == "__main__":
    main()