import os
import subprocess
import argparse
import yaml
import logging
from typing import Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_fixed
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> List[Dict[str, any]]:
    """
    Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML file.

    Returns:
        List[Dict[str, any]]: List of configuration dictionaries.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: If parsing the YAML file fails.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            return config if config else []
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def run_pipeline_script(script_path: str, args: List[str] = None) -> None:
    """
    Execute a script as a subprocess with optional arguments and retry logic.

    Args:
        script_path (str): Path to the script to execute.
        args (List[str], optional): List of command-line arguments.

    Raises:
        subprocess.CalledProcessError: If the script execution fails after retries.
    """
    try:
        cmd = ['python', script_path]
        if args:
            cmd.extend(args)
        subprocess.run(cmd, check=True)
        logger.info(f"Successfully executed {script_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to execute {script_path}: {e}")
        raise

def get_dataset_local_dir(dataset_id: str, datasets_config: List[Dict[str, str]]) -> Optional[str]:
    """
    Retrieve the local directory for a dataset from datasets_info.yaml.

    Args:
        dataset_id (str): Name of the dataset (e.g., CLAPv2/MusicCaps).
        datasets_config (List[Dict[str, str]]): Dataset configurations.

    Returns:
        Optional[str]: Local directory path or None if not found.
    """
    for dataset in datasets_config:
        if dataset['dataset_id'] == dataset_id:
            return dataset['local_dir']
    logger.warning(f"Dataset {dataset_id} not found in datasets_info.yaml")
    return None

def get_dataset_processed_data_dir(dataset_id: str, 
                                   datasets_config: List[Dict[str, str]]) -> Optional[str]:
    """
    Retrieve the local directory for a dataset from datasets_info.yaml.

    Args:
        dataset_id (str): Name of the dataset (e.g., CLAPv2/MusicCaps).
        datasets_config (List[Dict[str, str]]): Dataset configurations.

    Returns:
        Optional[str]: Local directory path or None if not found.
    """
    for dataset in datasets_config:
        if dataset['dataset_id'] == dataset_id:
            return dataset['processed_data_dir']
    logger.warning(f"Dataset {dataset_id} not found in process_datasets.yaml")
    return None

def get_checkpoint_info(model_id: str, ckpts_config: List[Dict[str, any]]) -> Optional[Dict[str, any]]:
    """
    Retrieve checkpoint information for a model from model_ckpts.yaml.

    Args:
        model_id (str): Model identifier.
        ckpts_config (List[Dict[str, any]]): Checkpoint configurations.

    Returns:
        Optional[Dict[str, any]]: Checkpoint configuration or None if not found.
    """
    for ckpt in ckpts_config:
        if ckpt['model_id'] == model_id:
            return ckpt
    logger.warning(f"Model {model_id} not found in model_ckpts.yaml")
    return None

def execute_pipeline(base_model_only: bool = False, finetune_only: bool = False,
                    model_id: Optional[str] = None, dataset_id: Optional[str] = None) -> None:
    """
    Execute the prerequisite pipeline scripts in sequence with retry logic.

    Args:
        base_model_only (bool): If True, download only base model checkpoints.
        finetune_only (bool): If True, download only fine-tuned model checkpoints.
        model_id (Optional[str]): Model identifier to target specific checkpoint downloads.
        dataset_id (Optional[str]): Dataset name to target specific dataset downloads.

    Raises:
        ValueError: If both base_model_only and finetune_only are True.
        Exception: For other execution errors.
    """
    if base_model_only and finetune_only:
        logger.error("Cannot specify both --base_model_only and --finetune_only")
        raise ValueError("Only one of --base_model_only or --finetune_only can be specified")

    try:
        # Step 1: Run setup_third_party.py
        run_pipeline_script(os.path.join('scripts', 'setup_third_party.py'))

        # Step 2: Run download_datasets.py with optional dataset_id
        dataset_args = ['--dataset_id', dataset_id] if dataset_id else []
        run_pipeline_script(os.path.join('scripts', 'download_datasets.py'), dataset_args)

        # Step 3: Run process_dataset.py with optional preset_name
        preset_args = []
        if model_id:
            preset_name = 'AudioLDM-finetuning' if 'AudioLDM' in model_id else 'MusicGen-Small-MusicCaps-finetuning'
            preset_args.extend(['--preset_name', preset_name])
        run_pipeline_script(os.path.join('scripts', 'process_dataset.py'), preset_args)

        # Step 4: Run download_ckpts.py with optional model_id and filters
        ckpt_args = []
        if base_model_only:
            ckpt_args.append('--base_model_only')
        elif finetune_only:
            ckpt_args.append('--finetune_only')
        if model_id:
            ckpt_args.extend(['--model_id', model_id])
        run_pipeline_script(os.path.join('scripts', 'download_ckpts.py'), ckpt_args)

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def train_model(preset_name: str, model_id: str, dataset_id: str, 
                third_party_dir: str, checkpoint_dir: str, 
                dataset_dir: str, output_model_dir: str) -> None:
    """
    Train a model using the specified dataset and third-party training script with retry logic.

    Args:
        preset_name (str): Name of the training preset.
        model_id (str): Identifier of the model to train.
        dataset_id (str): Dataset to use for training (e.g., CLAPv2/MusicCaps).
        third_party_dir (str): Directory of the cloned third-party repository.
        checkpoint_dir (str): Directory containing model checkpoints.
        dataset_dir (str): Directory containing processed dataset.
        output_model_dir (str): Directory to save the trained model.

    Raises:
        FileNotFoundError: If required paths do not exist.
        ValueError: If the preset or model is unsupported.
        subprocess.CalledProcessError: If the training command fails after retries.
    """
    try:
        dataset_path = os.path.join(dataset_dir)
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset path {dataset_path} does not exist")
            raise FileNotFoundError(f"Dataset path {dataset_path} not found")

        if not os.path.exists(checkpoint_dir):
            logger.error(f"Checkpoint directory {checkpoint_dir} does not exist")
            raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} not found")

        if preset_name == 'AudioLDM-finetuning':
            device = 'gpu' if torch.cuda.is_available() else 'cpu'
            train_cmd = [
                'python', os.path.join(third_party_dir, 'src', 'audioldm', 'train.py'),
                '--config_yaml', os.path.join('configs', 'AudioLDM_training_configs', 'audioldm_original.yaml'),
                '--reload_from_ckpt', os.path.join(checkpoint_dir, 'audioldm-s-full.ckpt'),
                '--wandb_off',
                '--accelerator', device,
                '--output_dir', output_model_dir
            ]
        elif preset_name == 'MusicGen-Small-MusicCaps-finetuning':
            train_cmd = [
                'python', os.path.join(third_party_dir, 'dreambooth_musicgen.py'),
                '--model_name_or_path', checkpoint_dir,
                '--is_resume_from_checkpoint', 'false',
                '--dataset_name', dataset_path,
                '--dataset_config_name', 'default',
                '--target_audio_column_name', 'audio',
                '--text_column_name', 'caption',
                '--train_split_name', 'train',
                '--eval_split_name', 'train',
                '--output_dir', output_model_dir,
                '--do_train',
                '--fp16',
                '--num_train_epochs', '700',
                '--gradient_accumulation_steps', '8',
                '--gradient_checkpointing',
                '--per_device_train_batch_size', '16',
                '--learning_rate', '1e-5',
                '--adam_beta1', '0.9',
                '--adam_beta2', '0.99',
                '--weight_decay', '0.1',
                '--guidance_scale', '3.0',
                '--do_eval',
                '--predict_with_generate',
                '--include_inputs_for_metrics',
                '--eval_steps', '25',
                '--per_device_eval_batch_size', '8',
                '--max_eval_samples', '64',
                '--generation_max_length', '400',
                '--dataloader_num_workers', '8',
                '--logging_steps', '1',
                '--max_duration_in_seconds', '30',
                '--min_duration_in_seconds', '1.0',
                '--preprocessing_num_workers', '4',
                '--pad_token_id', '2048',
                '--decoder_start_token_id', '2048',
                '--seed', '456',
                '--overwrite_output_dir',
                '--push_to_hub', 'false',
                '--save_total_limit', '1',
                '--report_to', 'none'
            ]
        else:
            logger.error(f"Unsupported preset_name: {preset_name}")
            raise ValueError(f"No training command defined for preset {preset_name}")

        logger.info(f"Training {model_id} with dataset {dataset_id} for preset {preset_name} at {dataset_path}, output to {output_model_dir}")
        subprocess.run(train_cmd, check=True)
        logger.info(f"Completed training for {model_id} with preset {preset_name}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed for {model_id} with preset {preset_name}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error training {model_id} with preset {preset_name}: {e}")
        raise

def main():
    """
    Main function to execute the pipeline and train a single model based on the selected preset.

    Raises:
        ValueError: For invalid arguments or missing configurations.
        Exception: For configuration, pipeline, or training errors.
    """
    config_path = os.path.join('configs', 'training_presets.yaml')
    process_datasets_config_path = os.path.join('configs', 'process_datasets.yaml')
    try:
        training_config = load_config(config_path)
        preset_choices = [preset.get('preset_name') for preset in training_config if preset.get('preset_name')]
        if not preset_choices:
            logger.error(f"No valid presets found in {config_path}")
            raise ValueError(f"No valid presets found in {config_path}")
    except Exception as e:
        logger.error(f"Failed to load preset choices from {config_path}: {e}")
        raise

    try:
        process_datasets_config = load_config(process_datasets_config_path)
    except Exception as e:
        logger.error(f"Failed to load preset choices from {process_datasets_config}: {e}")
        raise

    parser = argparse.ArgumentParser(description="Train a single text-to-music model based on the selected preset.")
    parser.add_argument(
        '--preset_name',
        type=str,
        choices=preset_choices,
        required=True,
        help=f'Name of the preset to train (choices: {", ".join(preset_choices)}).'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default=config_path,
        help='Path to the training presets YAML configuration file'
    )
    parser.add_argument(
        '--base_model_only',
        action='store_true',
        help='Download and train only base models'
    )
    parser.add_argument(
        '--finetune_only',
        action='store_true',
        help='Download and train only fine-tuned models'
    )
    parser.add_argument(
        '--both',
        action='store_true',
        help='Download and train both base and fine-tuned models (default)'
    )
    args = parser.parse_args()

    if sum([args.base_model_only, args.finetune_only, args.both]) > 1:
        logger.error("Only one of --base_model_only, --finetune_only, or --both can be specified")
        raise ValueError("Only one of --base_model_only, --finetune_only, or --both can be specified")

    if not args.base_model_only and not args.finetune_only and not args.both:
        args.both = True
        logger.info("No model filter specified; allowing both base and fine-tuned models")

    try:
        training_config = load_config(args.config_path)
        datasets_config = load_config(os.path.join('configs', 'datasets_info.yaml'))
        ckpts_config = load_config(os.path.join('configs', 'model_ckpts.yaml'))
    except Exception as e:
        logger.error(f"Failed to load configurations: {e}")
        raise
    
    # Training Preset
    training_config = [preset for preset in training_config 
                       if preset.get('preset_name').lower() == args.preset_name.lower()]
    if not training_config:
        logger.error(f"No preset found with name {args.preset_name}. Available presets: {preset_choices}")
        raise ValueError(f"Preset {args.preset_name} not found in {args.config_path}")
    
    # Process Datasets Config
    process_datasets_config = [preset for preset in process_datasets_config 
                       if preset.get('preset_name').lower() == args.preset_name.lower()]
    preset_process_datasets= process_datasets_config[0]
    # preset_name_process_datasets = preset_process_datasets.get('preset_name')
    processing_classs_process_datasets = preset_process_datasets.get('processing_classs')

    preset = training_config[0]
    preset_name = preset.get('preset_name')
    model_id = preset.get('model_id')
    dataset_id = preset.get('dataset_id')
    third_party_repo = preset.get('third_party')
    output_model_dir = preset.get('output_model_dir')

    if not all([preset_name, model_id, dataset_id, third_party_repo, output_model_dir]):
        logger.error(f"Incomplete preset: {preset}")
        raise ValueError(f"Preset {preset_name} is missing required fields")

    execute_pipeline(base_model_only=args.base_model_only, finetune_only=args.finetune_only,
                    model_id=model_id, dataset_id=dataset_id)

    ckpt_info = get_checkpoint_info(model_id, ckpts_config)
    if not ckpt_info:
        logger.error(f"No checkpoint information found for {model_id} in preset {preset_name}")
        raise ValueError(f"No checkpoint information found for {model_id}")

    is_base_model = ckpt_info.get('base_model', False)
    checkpoint_dir = ckpt_info.get('local_dir')

    if args.base_model_only and not is_base_model:
        logger.error(f"Model {model_id} for preset {preset_name} is not a base model")
        raise ValueError(f"Model {model_id} is not a base model")
    if args.finetune_only and is_base_model:
        logger.error(f"Model {model_id} for preset {preset_name} is not a fine-tuned model")
        raise ValueError(f"Model {model_id} is not a fine-tuned model")

    if processing_classs_process_datasets:
        dataset_dir = get_dataset_processed_data_dir(dataset_id, process_datasets_config)
    else:
        dataset_dir = get_dataset_local_dir(dataset_id, datasets_config)
    print("dataset_dir, dataset_dir, dataset_dir: ", dataset_dir)
    if not dataset_dir:
        logger.error(f"Dataset {dataset_id} local directory not found for preset {preset_name}")
        raise ValueError(f"Dataset {dataset_id} local directory not found")

    clone_dir = os.path.basename(third_party_repo).replace('.git', '')
    third_party_dir = os.path.join('src', 'third_party', clone_dir)

    train_model(preset_name, model_id, dataset_id, third_party_dir, 
                checkpoint_dir, dataset_dir, output_model_dir)

if __name__ == "__main__":
    main()