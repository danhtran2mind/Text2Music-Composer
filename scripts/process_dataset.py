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
from tqdm import tqdm  # Import tqdm for progress bars

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> List[Dict[str, str]]:
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        List[Dict[str, str]]: List of configurations.

    Raises:
        Exception: If loading the configuration fails.
    """
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise

def get_local_dir(dataset_id: str, datasets_info: List[Dict[str, str]]) -> str:
    """
    Get local_dir for a dataset from datasets_info.yaml.

    Args:
        dataset_id (str): Dataset ID to look up.
        datasets_info (List[Dict[str, str]]): List of dataset info configurations.

    Returns:
        str: Local directory path.

    Raises:
        ValueError: If dataset_id is not found in datasets_info.
    """
    for info in datasets_info:
        if info['dataset_id'] == dataset_id:
            return info['local_dir']
    raise ValueError(f"Dataset ID {dataset_id} not found in datasets_info.yaml")

def load_and_clean_dataset(dataset_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and clean the dataset, removing duplicates by keeping the row with the longest main_caption per location.

    Args:
        dataset_id (str): Hugging Face dataset name.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Cleaned train and validation DataFrames.

    Raises:
        ValueError: If dataset splits are missing.
        Exception: For other loading or cleaning errors.
    """
    try:
        dataset = load_dataset(dataset_id)
        if "train" not in dataset or "test" not in dataset:
            raise ValueError(f"Dataset {dataset_id} does not contain 'train' or 'test' splits.")
        
        train_df = pd.DataFrame(dataset["train"]).groupby("location")["main_caption"].apply(
            lambda x: x.loc[x.str.len().idxmax()]
        ).reset_index()
        val_df = pd.DataFrame(dataset["test"]).groupby("location")["main_caption"].apply(
            lambda x: x.loc[x.str.len().idxmax()]
        ).reset_index()
        
        logger.info(f"Loaded dataset {dataset_id}. Train set size: {len(train_df)}, Validation set size: {len(val_df)}")
        return train_df, val_df
    except Exception as e:
        logger.error(f"Failed to load or clean dataset {dataset_id}: {e}")
        raise

def extract_dataset(dataset_id: str, local_dir: str, music_bench_dir: str) -> None:
    """
    Download and extract the dataset from Hugging Face.

    Args:
        dataset_id (str): Hugging Face dataset name.
        local_dir (str): Directory to store raw downloaded data.
        music_bench_dir (str): Directory to extract the dataset.

    Raises:
        FileNotFoundError: If the tar file is not found.
        Exception: For other download or extraction errors.
    """
    try:
        # os.makedirs(music_bench_dir, exist_ok=True)
        # snapshot_download(repo_id=dataset_id, local_dir=local_dir, repo_type="dataset")
        
        tar_path = os.path.join(local_dir, "MusicBench.tar.gz")
        if not os.path.exists(tar_path):
            raise FileNotFoundError(f"Tar file not found: {tar_path}")
        
        with tarfile.open(tar_path, "r:gz") as tar:
            # Get list of members for progress bar
            members = tar.getmembers()
            # Add tqdm progress bar for extraction with fixed width and smooth updates
            for member in tqdm(members, desc="Extracting dataset", unit="file", leave=True):
                tar.extract(member, path=music_bench_dir)
        logger.info(f"Dataset {dataset_id} extracted to {music_bench_dir}")
    except Exception as e:
        logger.error(f"Failed to download or extract dataset {dataset_id}: {e}")
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
    processed_data_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Move files from datashare to processed directory, organize into train/test folders, and clean up.

    Args:
        local_dir (str): Directory containing raw downloaded data.
        music_bench_dir (str): Directory containing extracted dataset.
        train_df (pd.DataFrame): Training DataFrame.
        val_df (pd.DataFrame): Validation DataFrame.
        num_processes (int): Number of processes for parallel file moving.
        processed_data_dir (str): Directory for processed data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Updated train and validation DataFrames.

    Raises:
        FileNotFoundError: If datashare directory is missing.
        Exception: For other processing errors.
    """
    try:
        train_dir = os.path.join(processed_data_dir, "audioset", "train")
        test_dir = os.path.join(processed_data_dir, "audioset", "test")
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

        # Process train tasks with tqdm progress bar
        train_results = []
        logger.info("Moving training files...")
        with mp.Pool(processes=num_processes) as pool:
            for result in tqdm(pool.imap(move_file, train_tasks), total=len(train_tasks), 
                               desc="Moving train files"):
                train_results.append(result)
            for index, success in train_results:
                if not success:
                    train_df = train_df.drop(index, errors="ignore")
        
        # Process validation tasks with tqdm progress bar
        val_results = []
        logger.info("Moving validation files...")
        with mp.Pool(processes=num_processes) as pool:
            for result in tqdm(pool.imap(move_file, val_tasks), total=len(val_tasks), 
                               desc="Moving validation files"):
                val_results.append(result)
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

        # Clean up local_dir and music_bench_dir
        for dir_path in [music_bench_dir, local_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path, ignore_errors=True)
                logger.info(f"Cleaned up directory: {dir_path}")
        
        return train_df, val_df
    except Exception as e:
        logger.error(f"Error in move_and_cleanup_files for {processed_data_dir}: {e}")
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

def write_json_files(train_data: List[Dict[str, str]], val_data: List[Dict[str, str]], processed_data_dir: str) -> None:
    """
    Write train and validation data to JSON files.

    Args:
        train_data (List[Dict[str, str]]): Training data for JSON.
        val_data (List[Dict[str, str]]): Validation data for JSON.
        processed_data_dir (str): Directory for processed data.

    Raises:
        Exception: If writing JSON files fails.
    """
    try:
        json_dir = os.path.join(processed_data_dir, "audioset")
        os.makedirs(json_dir, exist_ok=True)
        
        with open(os.path.join(json_dir, "train.json"), "w") as f:
            json.dump({"data": train_data}, f, indent=4)
        with open(os.path.join(json_dir, "test.json"), "w") as f:
            json.dump({"data": val_data}, f, indent=4)
        logger.info(f"JSON files written to {json_dir}")
    except Exception as e:
        logger.error(f"Error writing JSON files for {processed_data_dir}: {e}")
        raise

def create_dataset_root_json(processed_data_dir: str) -> None:
    """
    Create dataset_root.json with metadata configuration.

    Args:
        processed_data_dir (str): Directory for processed data.

    Raises:
        Exception: If writing dataset_root.json fails.
    """
    try:
        dataset_root = {
            "audiocaps": f"./{processed_data_dir}/audioset",
            "comments": {},
            "metadata": {
                "path": {
                    "audiocaps": {
                        "train": f"./{processed_data_dir}/audioset/train.json",
                        "test": f"./{processed_data_dir}/audioset/test.json",
                        "class_label_indices": "../metadata/audiocaps/class_labels_indices.csv"
                    }
                }
            }
        }
        metadata_dir = os.path.join(processed_data_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        
        with open(os.path.join(metadata_dir, "dataset_root.json"), "w") as f:
            json.dump(dataset_root, f, indent=4)
        logger.info(f"Dataset root JSON created at {metadata_dir}")
    except Exception as e:
        logger.error(f"Error creating dataset_root.json for {processed_data_dir}: {e}")
        raise

def process_dataset(dataset_info: Dict[str, str], datasets_info: List[Dict[str, str]], num_processes: int) -> None:
    """
    Process a single dataset: load, clean, download, extract, move files, and create JSON files.

    Args:
        dataset_info (Dict[str, str]): Dataset configuration from process_datasets.yaml.
        datasets_info (List[Dict[str, str]]): Dataset info from datasets_info.yaml.
        num_processes (int): Number of processes for parallel file operations.

    Raises:
        Exception: For any processing errors.
    """
    preset_name = dataset_info['preset_name']
    dataset_id = dataset_info['dataset_id']
    processed_data_dir = dataset_info['processed_data_dir']
    processing_class = dataset_info.get('processing_classs', None)
    
    # Get local_dir from datasets_info.yaml
    local_dir = get_local_dir(dataset_id, datasets_info)
    music_bench_dir = os.path.join(local_dir, "audioset", "music_bench")

    try:
        logger.info(f"Processing dataset {preset_name} (ID: {dataset_id})")
        train_df, val_df = load_and_clean_dataset(dataset_id)
        
        # Apply custom processing class if specified
        if processing_class:
            logger.info(f"Applying custom processing class: {processing_class}")
            # Note: No implementation for processing_classs (e.g., 'abc') exists.
            logger.warning(f"Processing class {processing_class} is not implemented in this script.")
        
        extract_dataset(dataset_id, local_dir, music_bench_dir)
        train_df, val_df = move_and_cleanup_files(local_dir, music_bench_dir, train_df, val_df, num_processes, processed_data_dir)
        train_data, val_data = prepare_json_data(train_df, val_df)
        write_json_files(train_data, val_data, processed_data_dir)
        create_dataset_root_json(processed_data_dir)
        logger.info(f"Successfully processed dataset {preset_name} (ID: {dataset_id})")
    except Exception as e:
        logger.error(f"Error processing dataset {preset_name} (ID: {dataset_id}): {e}")
        raise

def main():
    """
    Main function to process datasets based on configuration.

    Raises:
        Exception: For configuration or processing errors.
    """
    parser = argparse.ArgumentParser(description="Process datasets from HuggingFace with parallel file operations.")
    parser.add_argument('--num_processes', type=int, default=os.cpu_count() or 1,
                        help='Number of processes to use (capped at CPU count)')
    parser.add_argument('--preset_name', type=str, default=True,
                        help='Name of the preset to process (optional; if not provided, process all presets)')
    args = parser.parse_args()

    # Validate number of processes
    num_processes = min(args.num_processes, os.cpu_count() or 1)
    logger.info(f"Using {num_processes} processes for parallel file operations.")

    # Load configurations
    process_config_path = os.path.join('configs', 'process_datasets.yaml')
    datasets_info_path = os.path.join('configs', 'datasets_info.yaml')
    try:
        process_config = load_config(process_config_path)
        datasets_info = load_config(datasets_info_path)
    except Exception as e:
        logger.error(f"Failed to load configurations: {e}")
        return

    # Filter datasets by preset_name if provided
    if args.preset_name:
        process_config = [dataset_info for dataset_info in process_config 
                         if dataset_info['preset_name'] == args.preset_name]
        if not process_config:
            logger.error(f"No dataset found with preset_name: {args.preset_name}")
            return

    # Process each dataset with tqdm progress bar
    for dataset_info in tqdm(process_config, desc="Processing datasets"):
        process_dataset(dataset_info, datasets_info, num_processes)

if __name__ == "__main__":
    main()