import sys
import os
import subprocess
import argparse
import yaml


def setup_third_party(config_path, src_path):
    
    # Load repository configurations from YAML file
    try:
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")

    # Extract repository configurations and derive clone directories
    repo_configs = []
    for config in config_data:
        repo_url = config.get('third_party')
        if repo_url:
            # Derive clone_dir from the repository URL (last part of the URL without .git)
            clone_dir = os.path.basename(repo_url).replace('.git', '')
            repo_configs.append({
                'third_party': repo_url,
                'clone_dir': clone_dir
            })

    # Validate that at least one repository is specified
    if not repo_configs:
        raise ValueError("No valid third-party repositories found in the configuration")

    # Append src/third_party to sys.path
    sys.path.append(src_path)

    # Create third_party directory if it doesn't exist
    os.makedirs(src_path, exist_ok=True)

    # Clone each repository into the specified directory
    for config in repo_configs:
        repo_url = config['third_party']
        clone_dir = config['clone_dir']
        clone_path = os.path.join(src_path, clone_dir)
        if not os.path.exists(clone_path):
            subprocess.run(['git', 'clone', repo_url, clone_path], check=True)
        else:
            print(f"Directory {clone_path} already exists, skipping clone for {repo_url}")


if __name__ == '__main__':
    # Define the argument parser for command-line inputs
    parser = argparse.ArgumentParser(description="Setup third-party dependencies based on training options configuration")
    parser.add_argument('--config_path', default=os.path.join('configs', 'training_presets.yaml'),
                        help='Path to the YAML configuration file')
    parser.add_argument('--src_path', default=os.path.join('src', 'third_party'),
                        help='Path to append to sys.path and clone repositories to')

    args = parser.parse_args()

    setup_third_party(args.config_path, args.src_path)