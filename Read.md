# NLP Project: ORPO Fine-Tuning for Language Models

This project implements Odds Ratio Preference Optimization (ORPO) for fine-tuning language models, focusing on preference alignment using binarized feedback datasets. It includes training scripts, evaluation pipelines, and model checkpoints for models like Mistral and Qwen2.5.

## Project Structure

- `main.py`: Main entry point for running training or evaluation.
- `pip_requirements.txt`: Python dependencies required for the project.
- `checkpoints/`: Saved model checkpoints from training runs.
  - `ultrafeedback-binarized-preferences-cleaned/`: Contains checkpoints for Qwen2.5-1.5B-Instruct fine-tuned with ORPO.
    - `Qwen2.5-1.5B-Instruct-ultrafeedback-binarized-preferences-cleaned-lambda1.0-ORPO-30-5-29/`: Main model with multiple checkpoint stages (checkpoint-325, checkpoint-650, checkpoint-975, etc.).
- `data/`: Training and evaluation datasets.
  - `train_first_10_samples.json`: Sample training data.
- `outputs/`: Evaluation results from trained models.
  - `alpacaeval/`: AlpacaEval benchmark results (Mistral-ORPO variants).
  - `mtbench/`: MTBench evaluation outputs (JSONL format).
- `results/`: Training logs and W&B export data.
  - `trainer_state.json`: Training state information.
  - `wandb_export_*.csv`: Exported W&B metrics and logs.
  - `wandb-summary.json`: Training summary.
- `src/`: Core source code.
  - `args.py`: Argument parsing utilities.
  - `orpo_trainer.py`: Custom ORPO trainer implementation.
  - `utils.py`: Utility functions.
- `trl/`: Test and demo files for the ORPO trainer.
  - `test_orpo_trainer_demo.py`: Demo script for testing ORPO trainer.
- `wandb/`: Weights & Biases logging data and run artifacts.

## Installation

1. Clone or download the project repository.
2. Install dependencies:
   ```
   pip install -r pip_requirements.txt
   ```
3. Ensure you have the necessary hardware (GPU recommended for training).
4. **Set up Weights & Biases (W&B) for logging**:
   - Create a W&B account at [https://wandb.ai](https://wandb.ai)
   - Obtain your API key from your W&B account settings
   - Authenticate by running: `wandb login` and enter your API key when prompted
   - This is required to log training metrics and results to W&B

## Usage

### Training

Run the main training script with appropriate configuration:

```bash
python main.py     --model_name Qwen/Qwen2.5-1.5B-Instruct     --data_name argilla/ultrafeedback-binarized-preferences-cleaned     --save_dir ./qwen2.5-1.5b-llm-success     --enable_lora     --num_proc 0     --lr 5e-6     --per_device_train_batch_size 1     --gradient_accumulation_steps 8     --max_train_samples 20
```

For custom configurations, modify `main.py` or pass command-line arguments as defined in `src/args.py`.

### Evaluation

Evaluation results are automatically saved to `outputs/` during training runs. Results include:
- AlpacaEval benchmark outputs in JSON format
- MTBench evaluation in JSONL format

Training metrics and logs are tracked with Weights & Biases and exported to `results/`.

## Model Checkpoints

Pre-trained checkpoints are available in `checkpoints/`:
- **Qwen2.5-1.5B-Instruct** with ORPO fine-tuning on UltraFeedback (binarized preferences)
- Multiple training stages: checkpoint-325, checkpoint-650, checkpoint-975, checkpoint-1300, checkpoint-1625, checkpoint-1950, checkpoint-2275, checkpoint-2600, checkpoint-2925, checkpoint-3250
- Each checkpoint includes adapter weights (LoRA format), tokenizer configs, and chat templates

Load them using the model loading utilities in `src/utils.py`.

## Dependencies

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- Accelerate
- Weights & Biases (for logging)
- Other libraries listed in `pip_requirements.txt`

## Results

Training and evaluation results are stored in dedicated directories:
- **`results/`**: Contains trainer state and Weights & Biases export data (CSV, JSON formats) with metrics tracked during training
- **`outputs/`**: Contains benchmark evaluation results for different model variants (AlpacaEval JSON and MTBench JSONL formats)
- **`checkpoints/`**: Each checkpoint directory includes trainer state, optimizer state, and scheduler information

## Contributing

Feel free to submit issues or pull requests for improvements.