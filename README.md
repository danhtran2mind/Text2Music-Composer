# Text2Music Composer 🎶








## Installation
### Clone GitHub Repository
```bash
git clone https://github.com/danhtran2mind/Anime-Super-Resolution
cd Anime-Super-Resolution
```
### Install Dependencies (Training + Inference)
```bash
pip install -e .
```
### Install Dependencies for Inference only
```bash
pip install -r requirements/requirements_inference.txt
```
### Script-Driven Setup

#### Download Model checkpoints

- Download only model checkpoint for Inference
    ```bash
    python scripts/download_ckpts.py --finetune_only
    ```

- Download all models (base and finetuned, as in the original behavior):
    ```bash
    python download_ckpts.py  --base_model_only
    ```
- Download all models (base and finetuned, as in the original behavior):
    ```bash
    python download_ckpts.py
    ```

More detail you can read at [Download Model Checkpoints](docs/scripts/download_model_ckpts.md).

#### Setup Third Party
```bash
    python scripts/setup_third_party.py
```

#### Download Dataset
```bash
    python scripts/download_datasets.py \
        --dataset_id "<huggingface_dataset_id>"
        --huggingface_token "<your_huggingface_token>"
```
More detail you can read at [Download Dataset](docs/scripts/download_dataset_doc.md).




Audio-finetuning

https://github.com/haoheliu/AudioLDM-training-finetuning

https://zenodo.org/records/14342967