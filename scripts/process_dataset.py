import argparse
import yaml
import os
import logging
import pandas as pd
from datasets import load_dataset
from huggingface_hub import snapshot_download
import shutil
import tarfile
import json
import multiprocessing as mp
from typing import Tuple, List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> List[Dict[str, str]]:
    """
    Load dataset configuration from YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        List[Dict[str, str]]: List of dataset configurations.

    Raises:
        Exception: If loading the configuration fails.
    """
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise

def load_and_clean_dataset(dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and clean the dataset, removing duplicates by keeping the row with the longest main_caption per location.

    Args:
        dataset_name (str): Hugging Face dataset name.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Cleaned train and validation DataFrames.

    Raises:
        ValueError: If dataset splits are missing.
        Exception: For other loading or cleaning errors.
    """
    try:
        dataset = load_dataset(dataset_name)
        if "train" not in dataset or "test" not in dataset:
            raise ValueError(f"Dataset {dataset_name} does not contain 'train' or 'test' splits.")
        
        train_df = pd.DataFrame(dataset["train"]).groupby("location")["main_caption"].apply(
            lambda x: x.loc[x.str.len().idxmax()]
        ).reset_index()
        val_df = pd.DataFrame(dataset["test"]).groupby("location")["main_caption"].apply(
            lambda x: x.loc[x.str.len().idxmax()]
        ).reset_index()
        
        logger.info(f"Loaded dataset {dataset_name}. Train set size: {len(train_df)}, Validation set size: {len(val_df)}")
        return train_df, val_df
    except Exception as e:
        logger.error(f"Failed to load or clean dataset {dataset_name}: {e}")
        raise

def download_and_extract_dataset(dataset_name: str, local_dir: str, music_bench_dir: str) -> None:
    """
    Download and extract the dataset from Hugging Face.

    Args:
        dataset_name (str): Hugging Face dataset name.
        local_dir (str): Directory to store raw downloaded data.
        music_bench_dir (str): Directory to extract the dataset.

    Raises:
        FileNotFoundError: If the tar file is not found.
        Exception: For other download or extraction errors.
    """
    try:
        os.makedirs(music_bench_dir, exist_ok=True)
        snapshot_download(repo_id=dataset_name, local_dir=local_dir, repo_type="dataset")
        
        tar_path = os.path.join(local_dir, "MusicBench.tar.gz")
        if not os.path.exists(tar_path):
            raise FileNotFoundError(f"Tar file not found: {tar_path}")
        
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=music_bench_dir)
        logger.info(f"Dataset {dataset_name} extracted to {music_bench_dir}")
    except Exception as e:
        logger.error(f"Failed to download or extract dataset {dataset_name}: {e}")
        raise

def move_file(args: Tuple[str, str, int]) -> Tuple[int, bool]:
    """
    Move a single file and handle errors.

    Args:
        args (Tuple[str, str, int]): Source path, destination path, and index.

    Returns:
        Tuple[int, bool]: Index and success status.
    """
    src_path, dst_path, index = args
    try:
        if os.path.exists(src_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.move(src_path, dst_path)
            return index, True
        else:
            logger.warning(f"File not found: {src_path}")
            return index, False
    except Exception as e:
        logger.error(f"Error moving file {src_path} to {dst_path}: {e}")
        return index, False

def move_and_cleanup_files(
    local_dir: str,
    music_bench_dir: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_processes: int,
    dataset_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Move files from datashare to music_bench, organize into train/test folders, and clean up.

    Args:
        local_dir (str): Directory containing raw downloaded data.
        music_bench_dir (str): Directory containing extracted dataset.
        train_df (pd.DataFrame): Training DataFrame.
        val_df (pd.DataFrame): Validation DataFrame.
        num_processes (int): Number of processes for parallel file moving.
        dataset_dir (str): Dataset directory name.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Updated train and validation DataFrames.

    Raises:
        FileNotFoundError: If datashare directory is missing.
        Exception: For other processing errors.
    """
    try:
        train_dir = os.path.join("data", dataset_dir, "audioset", "train")
        test_dir = os.path.join("data", dataset_dir, "audioset", "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        datashare_dir = os.path.join(music_bench_dir, "datashare")
        if not os.path.exists(datashare_dir):
            raise FileNotFoundError(f"Datashare directory not found: {datashare_dir}")

        train_tasks = [
            (os.path.join(datashare_dir, row["location"]),
             os.path.join(train_dir, os.path.basename(row["location"])),
             index)
            for index, row in train_df.iterrows()
        ]
        val_tasks = [
            (os.path.join(datashare_dir, row["location"]),
             os.path.join(test_dir, os.path.basename(row["location"])),
             index)
            for index, row in val_df.iterrows()
        ]

        with mp.Pool(processes=num_processes) as pool:
            train_results = pool.map(move_file, train_tasks)
            for index, success in train_results:
                if not success:
                    train_df = train_df.drop(index, errors="ignore")
            
            val_results = pool.map(move_file, val_tasks)
            for index, success in val_results:
                if not success:
                    val_df = val_df.drop(index, errors="ignore")

        train_filenames = set(os.path.basename(row["location"]) for _, row in train_df.iterrows())
        for filename in os.listdir(train_dir):
            if filename not in train_filenames:
                file_path = os.path.join(train_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed extra file from train folder: {file_path}")

        test_filenames = set(os.path.basename(row["location"]) for _, row in val_df.iterrows())
        for filename in os.listdir(test_dir):
            if filename not in test_filenames:
                file_path = os.path.join(test_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed extra file from test folder: {file_path}")

        for dir_path in [music_bench_dir, local_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path, ignore_errors=True)
                logger.info(f"Cleaned up directory: {dir_path}")
        
        return train_df, val_df
    except Exception as e:
        logger.error(f"Error in move_and_cleanup_files for {dataset_dir}: {e}")
        raise

def prepare_json_data(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Prepare JSON data for train and validation sets with updated paths.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        val_df (pd.DataFrame): Validation DataFrame.

    Returns:
        Tuple[List[Dict[str, str]], List[Dict[str, str]]]: Train and validation data for JSON.

    Raises:
        Exception: If preparing JSON data fails.
    """
    try:
        train_data = [
            {"wav": os.path.join("train", os.path.basename(row["location"])),
             "caption": row["main_caption"]}
            for _, row in train_df.iterrows()
        ]
        val_data = [
            {"wav": os.path.join("test", os.path.basename(row["location"])),
             "caption": row["main_caption"]}
            for _, row in val_df.iterrows()
        ]
        return train_data, val_data
    except Exception as e:
        logger.error(f"Error preparing JSON data: {e}")
        raise

def write_json_files(train_data: List[Dict[str, str]], val_data: List[Dict[str, str]], dataset_dir: str) -> None:
    """
    Write train and validation data to JSON files.

    Args:
        train_data (List[Dict[str, str]]): Training data for JSON.
        val_data (List[Dict[str, str]]): Validation data for JSON.
        dataset_dir (str): Dataset directory name.

    Raises:
        Exception: If writing JSON files fails.
    """
    try:
        json_dir = os.path.join("data", dataset_dir, "audioset")
        os.makedirs(json_dir, exist_ok=True)
        
        with open(os.path.join(json_dir, "train.json"), "w") as f:
            json.dump({"data": train_data}, f, indent=4)
        with open(os.path.join(json_dir, "test.json"), "w") as f:
            json.dump({"data": val_data}, f, indent=4)
        logger.info(f"JSON files written to {json_dir}")
    except Exception as e:
        logger.error(f"Error writing JSON files for {dataset_dir}: {e}")
        raise

def create_dataset_root_json(dataset_dir: str) -> None:
    """
    Create dataset_root.json with metadata configuration.

    Args:
        dataset_dir (str): Dataset directory name.

    Raises:
        Exception: If writing dataset_root.json fails.
    """
    try:
        dataset_root = {
            "audiocaps": f"./data/{dataset_dir}/audioset",
            "comments": {},
            "metadata": {
                "path": {
                    "audiocaps": {
                        "train": f"./data/{dataset_dir}/audioset/train.json",
                        "test": f"./data/{dataset_dir}/audioset/test.json",
                        "class_label_indices": "../metadata/audiocaps/class_labels_indices.csv"
                    }
                }
            }
        }
        metadata_dir = os.path.join("data", dataset_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        
        with open(os.path.join(metadata_dir, "dataset_root.json"), "w") as f:
            json.dump(dataset_root, f, indent=4)
        logger.info(f"Dataset root JSON created at {metadata_dir}")
    except Exception as e:
        logger.error(f"Error creating dataset_root.json for {dataset_dir}: {e}")
        raise

def process_dataset(dataset_info: Dict[str, str], local_data_dir: str, num_processes: int) -> None:
    """
    Process a single dataset: load, clean, download, extract, move files, and create JSON files.

    Args:
        dataset_info (Dict[str, str]): Dataset configuration (name and local directory).
        local_data_dir (str): Base directory for raw and processed data.
        num_processes (int): Number of processes for parallel file operations.

    Raises:
        Exception: For any processing errors.
    """
    dataset_name = dataset_info['dataset_name']
    dataset_dir = dataset_info['local_dir']
    local_dir = os.path.join(local_data_dir, f"temp--{dataset_dir}")
    
    # Change existed data_dir to temp if temp not exists
    if not os.path.exists(local_dir):
        os.rename(os.path.join(local_data_dir, dataset_dir), local_dir)

    music_bench_dir = os.path.join(local_data_dir, dataset_dir, "audioset", "music_bench")

    try:
        logger.info(f"Processing dataset {dataset_name}")
        train_df, val_df = load_and_clean_dataset(dataset_name)
        download_and_extract_dataset(dataset_name, local_dir, music_bench_dir)
        train_df, val_df = move_and_cleanup_files(local_dir, music_bench_dir, train_df, val_df, num_processes, dataset_dir)
        train_data, val_data = prepare_json_data(train_df, val_df)
        write_json_files(train_data, val_data, dataset_dir)
        create_dataset_root_json(dataset_dir)
        logger.info(f"Successfully processed dataset {dataset_name}")
    except Exception as e:
        logger.error(f"Error processing dataset {dataset_name}: {e}")
        raise

def main():
    """
    Main function to process datasets based on configuration.

    Raises:
        Exception: For configuration or processing errors.
    """
    parser = argparse.ArgumentParser(description="Process datasets from HuggingFace with parallel file operations.")
    parser.add_argument('--local_dir', type=str, default='./data',
                        help='Local directory to save datasets')
    parser.add_argument('--num_processes', type=int, default=os.cpu_count() or 1,
                        help='Number of processes to use (capped at CPU count)')
    args = parser.parse_args()

    # Validate number of processes
    num_processes = min(args.num_processes, os.cpu_count() or 1)
    logger.info(f"Using {num_processes} processes for parallel file operations.")

    # Load configuration
    config_path = os.path.join('configs', 'datasets_info.yaml')
    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    # Process each dataset
    for dataset_info in config:
        process_dataset(dataset_info, args.local_dir, num_processes)

if __name__ == "__main__":
    main()