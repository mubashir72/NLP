# NLP Project: ORPO Fine-Tuning for Language Models

This project implements Odds Ratio Preference Optimization (ORPO) for fine-tuning language models, focusing on preference alignment using binarized feedback datasets. It includes training scripts, evaluation pipelines, and model checkpoints for models like Mistral and Qwen2.5.

## Project Structure

- `main.py`: Main entry point for running training or evaluation.
- `pip_requirements.txt`: Python dependencies required for the project.
- `assets/`: Static assets, including images.
- `checkpoints/`: Saved model checkpoints from training runs.
  - `ultrafeedback-binarized-preferences-cleaned/`: Checkpoints for Qwen2.5-1.5B-Instruct fine-tuned with ORPO.
- `outputs/`: Evaluation results.
  - `alpacaeval/`: AlpacaEval results for different model variants.
  - `mtbench/`: MTBench evaluation outputs.
- `scripts/`: Shell scripts for running specific training configurations.
  - `run_mistral_orpo_beta.sh`: Run Mistral ORPO with beta configuration.
  - `run_mistral_orpo_capybara.sh`: Run Mistral ORPO with Capybara dataset.
  - `run_orpo.sh`: General ORPO training script.
- `src/`: Core source code.
  - `args.py`: Argument parsing utilities.
  - `orpo_trainer.py`: Custom ORPO trainer implementation.
  - `utils.py`: Utility functions.
  - `accelerate/`: Configuration files for distributed training (DeepSpeed and FSDP).
- `trl/`: Test files for the ORPO trainer.
- `wandb/`: Weights & Biases logging data from training runs.

## Installation

1. Clone or download the project repository.
2. Install dependencies:
   ```
   pip install -r pip_requirements.txt
   ```
3. Ensure you have the necessary hardware (GPU recommended for training).

## Usage

### Training

Use the provided scripts to run training:

- For general ORPO training: `./scripts/run_orpo.sh`
- For Mistral with beta config: `./scripts/run_mistral_orpo_beta.sh`
- For Mistral with Capybara: `./scripts/run_mistral_orpo_capybara.sh`

Alternatively, run directly with Python:
```
python main.py --config <config_file> --model <model_name>
```

### Evaluation

Evaluation results are stored in `outputs/`. Use the scripts or modify `main.py` to run evaluations on AlpacaEval or MTBench.

## Model Checkpoints

Pre-trained checkpoints are available in `checkpoints/`. Load them using the model loading utilities in `src/utils.py`.

## Dependencies

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- Accelerate
- Weights & Biases (for logging)
- Other libraries listed in `pip_requirements.txt`

## Results

Evaluation results show performance on AlpacaEval and MTBench for different ORPO variants. Refer to `outputs/` for detailed JSON/JSONL files.

## Contributing

Feel free to submit issues or pull requests for improvements.

## License

[Specify license if applicable, e.g., MIT]