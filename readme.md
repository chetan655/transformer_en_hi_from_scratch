# Transformer-based Neural Machine Translation

This project implements a Transformer model from scratch in PyTorch for bilingual (English-Hindi) machine translation. It includes data preprocessing, tokenization, model architecture, training, and validation.

## Features

- Custom Transformer architecture (encoder-decoder)
- PyTorch-based training loop
- Tokenization using HuggingFace `tokenizers`
- TensorBoard logging
- Model checkpointing
- Greedy decoding for inference

## Project Structure

```
.
├── config.py         # Configuration and utility functions
├── dataset.py        # Dataset and mask utilities
├── model.py          # Transformer model and components
├── train.py          # Training and validation loop
├── filtered5_iitb_en_hi.jsonl  # Example dataset (JSONL format)
└── weights/          # Model checkpoints (created after training)
```

## Requirements

- Python 3.8+
- torch
- tokenizers
- datasets
- tqdm
- tensorboard

Install dependencies with:

```bash
pip install torch tokenizers datasets tqdm tensorboard
```

## Data

The project expects a JSONL file with bilingual sentence pairs, e.g.:

```json
{ "translation": { "en": "Hello", "hi": "नमस्ते" } }
```

Update the data file path in `train.py` as needed.

## Training

To train the model, run:

```bash
python train.py
```

Model checkpoints will be saved in the `weights/` directory.

## Monitoring

You can monitor training loss using TensorBoard:

```bash
tensorboard --logdir runs/
```

## Inference

After training, you can use the model and tokenizer to translate new sentences using the `greedy_decode` function in `train.py`.

## Configuration

Edit `config.py` to change hyperparameters, file paths, or training settings.

## Acknowledgements

- Based on the original Transformer paper: ["Attention is All You Need"](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- Uses HuggingFace `tokenizers` and `datasets`

---

**Author:** Chetan Sharma  
**License:** MIT
