import yaml
import os
import argparse

def update_audioldm_config(training_presets_path='configs/training_presets.yaml', 
                          model_ckpts_path='configs/model_ckpts.yaml', 
                          audioldm_config_path='configs/AudioLDM_training_configs/audioldm_original.yaml', 
                          preset_name='AudioLDM-finetuning'):
    # Load the training presets YAML
    with open(training_presets_path, 'r') as file:
        training_presets = yaml.safe_load(file)

    # Load the model checkpoints YAML
    with open(model_ckpts_path, 'r') as file:
        model_ckpts = yaml.safe_load(file)

    # Find the specified preset
    preset = next((p for p in training_presets if p['preset_name'] == preset_name), None)
    if not preset:
        raise ValueError(f"Preset '{preset_name}' not found in {training_presets_path}")

    # Find the corresponding model checkpoint entry
    model_ckpt = next((m for m in model_ckpts if m['model_id'] == preset['model_id']), None)
    if not model_ckpt:
        raise ValueError(f"Model ID {preset['model_id']} not found in {model_ckpts_path}")

    local_dir = model_ckpt['local_dir']
    processed_data_dir = preset['processed_data_dir']

    # Load the AudioLDM config
    with open(audioldm_config_path, 'r') as file:
        audioldm_config = yaml.safe_load(file)

    # Update the specified fields
    audioldm_config['metadata_root'] = f"{processed_data_dir}/metadata/dataset_root.json"
    audioldm_config['log_directory'] = f"{local_dir}/log/latent_diffusion"
    audioldm_config['model']['params']['first_stage_config']['params']['reload_from_ckpt'] = f"{local_dir}/vae_mel_16k_64bins.ckpt"
    audioldm_config['model']['params']['cond_stage_config']['film_clap_cond1']['params']['pretrained_path'] = f"{local_dir}/clap_htsat_tiny.pt"

    # Save the updated AudioLDM config
    with open(audioldm_config_path, 'w') as file:
        yaml.safe_dump(audioldm_config, file, default_flow_style=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update AudioLDM configuration file.")
    parser.add_argument('--preset_name', type=str, default='AudioLDM-finetuning', 
                        help="Preset name to process (default: AudioLDM-finetuning)")
    parser.add_argument('--training_presets_path', type=str, default='configs/training_presets.yaml', 
                        help="Path to training presets YAML file")
    parser.add_argument('--model_ckpts_path', type=str, default='configs/model_ckpts.yaml', 
                        help="Path to model checkpoints YAML file")
    parser.add_argument('--audioldm_config_path', type=str, 
                        default='configs/AudioLDM_training_configs/audioldm_original.yaml', 
                        help="Path to AudioLDM configuration YAML file")
    args = parser.parse_args()

    update_audioldm_config(args.training_presets_path, args.model_ckpts_path, 
                           args.audioldm_config_path, args.preset_name)
    print(f"Updated {args.audioldm_config_path} successfully for preset '{args.preset_name}'.")