# Hibiscus

Transformer architecture built from scratch in PyTorch.

```
                            ┌─────────────────────────────────────┐
                            │           TRANSFORMER               │
                            │                                     │
   Input                    │  ┌─────────┐      ┌─────────┐      │    Output
  Sequence                  │  │ ENCODER │      │ DECODER │      │   Sequence
     │                      │  │         │      │         │      │      ▲
     ▼                      │  │  ┌───┐  │      │  ┌───┐  │      │      │
┌─────────┐                 │  │  │Att│  │      │  │Att│  │      │      │
│ Embed + │─────────────────┼─►│  └─┬─┘  │─────►│  └─┬─┘  │──────┼──────┘
│   Pos   │                 │  │    │    │      │    │    │      │
└─────────┘                 │  │  ┌─▼─┐  │      │  ┌─▼─┐  │      │
                            │  │  │FFN│  │      │  │FFN│  │      │
                            │  │  └───┘  │      │  └───┘  │      │
                            │  │   x6    │      │   x6    │      │
                            │  └─────────┘      └─────────┘      │
                            └─────────────────────────────────────┘
```

## What's inside

- Multi-head self-attention
- Positional encoding (sinusoidal + learned)
- BPE / WordPiece tokenizer
- Mixed precision training (AMP)
- Multi-GPU support (DDP)
- TensorBoard + W&B logging

## Setup

```bash
git clone https://github.com/prashanth8983/hibiscus.git
cd hibiscus
pip install -e .
```

## Train a model

```bash
# Small model (12M params)
python scripts/train.py --config configs/small.yaml

# With sample data
python scripts/train.py --config configs/small.yaml --create-sample-data
```

## Use in code

```python
from transformer import Transformer, ModelConfig

config = ModelConfig(
    vocab_size=30000,
    d_model=512,
    n_heads=8,
    n_layers=6
)

model = Transformer(config)
output = model(input_ids, target_ids)
```

## Generate text

```python
generated = model.generate(
    prompt_ids,
    max_len=100,
    temperature=0.8,
    top_k=10
)
```

## Model sizes

```
┌──────────┬────────────┬─────────┬─────────┬────────┐
│  Model   │   Params   │ d_model │ n_heads │ layers │
├──────────┼────────────┼─────────┼─────────┼────────┤
│  Small   │    12M     │   512   │    8    │   6    │
│  Medium  │    85M     │   768   │   12    │   12   │
│  Large   │   355M     │  1024   │   16    │   24   │
└──────────┴────────────┴─────────┴─────────┴────────┘
```

## Project layout

```
hibiscus/
├── transformer/
│   ├── model.py          # Main architecture
│   ├── attention.py      # Multi-head attention
│   ├── trainer.py        # Training loop
│   ├── tokenizer.py      # BPE/WordPiece
│   └── config.py         # YAML config loader
├── configs/
│   ├── small.yaml
│   ├── medium.yaml
│   └── large.yaml
└── scripts/
    ├── train.py
    └── evaluate.py
```

## Config example

```yaml
model:
  vocab_size: 30000
  d_model: 512
  n_heads: 8
  n_layers: 6
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 1e-4
  warmup_steps: 4000
  use_amp: true
```

## Monitor training

```bash
tensorboard --logdir logs/
```

## License

MIT
