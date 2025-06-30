# Professional Production-Grade Transformer

A comprehensive, production-ready implementation of the Transformer architecture with modern best practices, extensive testing, and professional tooling.

## 🚀 Features

### Core Architecture
- **Complete Transformer Implementation**: Full encoder-decoder architecture with multi-head attention
- **Advanced Attention Mechanisms**: Scaled dot-product attention, self-attention, cross-attention
- **Multiple Positional Encodings**: Sinusoidal and learned positional encodings
- **Flexible Feed-Forward Networks**: Configurable activation functions (ReLU, GELU, Swish)
- **Layer Normalization**: Proper residual connections and normalization

### Training Infrastructure
- **Professional Training Loop**: Comprehensive training with mixed precision, gradient clipping
- **Advanced Optimizers**: Adam, AdamW, SGD with configurable parameters
- **Learning Rate Scheduling**: Warmup + cosine annealing, linear decay, step decay
- **Mixed Precision Training**: Automatic mixed precision (AMP) support
- **Distributed Training**: Multi-GPU training with DDP support

### Data Processing
- **Multiple Tokenizers**: BPE, WordPiece, Character-level tokenization
- **Flexible Data Loading**: Text datasets, translation datasets with proper batching
- **Data Augmentation**: Configurable augmentation strategies
- **Efficient Preprocessing**: Parallel data loading with proper collation

### Configuration Management
- **YAML Configuration**: Hierarchical configuration system
- **Pre-configured Models**: Small (12M), Medium (85M), Large (355M) parameter variants
- **Environment Management**: Easy switching between different model sizes
- **Validation**: Comprehensive parameter validation and error checking

### Monitoring & Logging
- **TensorBoard Integration**: Real-time training visualization
- **Weights & Biases**: Experiment tracking and model versioning
- **Comprehensive Logging**: Training metrics, validation results, model statistics
- **Checkpointing**: Automatic model saving and resuming

### Analysis & Evaluation
- **Model Analysis**: Parameter statistics, complexity analysis, performance metrics
- **Attention Visualization**: Heatmap visualization of attention patterns
- **Performance Benchmarking**: Inference speed, memory usage analysis
- **Text Generation**: Configurable sampling strategies (temperature, top-k, top-p)

### Testing & Quality Assurance
- **Comprehensive Test Suite**: Unit tests for all components
- **Integration Tests**: End-to-end pipeline testing
- **Performance Tests**: Benchmarking and regression testing
- **Code Quality**: Type hints, linting, formatting

## 📦 Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Quick Install
```bash
# Clone the repository
git clone <repository-url>
cd Hibiscus

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Install pre-commit hooks for code quality
pre-commit install
```

### Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.62.0
pyyaml>=6.0
wandb>=0.13.0
tensorboard>=2.10.0
datasets>=2.0.0
tokenizers>=0.13.0
transformers>=4.20.0
scikit-learn>=1.1.0
pytest>=7.0.0
black>=22.0.0
flake8>=5.0.0
mypy>=0.991
pre-commit>=2.20.0
```

## 🎯 Quick Start

### Basic Usage
```python
from transformer import Transformer, ModelConfig

# Create configuration
config = ModelConfig(
    vocab_size=30000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    max_seq_len=512,
    dropout=0.1
)

# Initialize model
model = Transformer(config)

# Forward pass
batch_size, seq_len = 32, 128
x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
output = model(x, x)  # For language modeling
```

### Training
```python
from transformer import Config, Trainer, TextDataset, Tokenizer
from transformer.data import create_data_loaders

# Load configuration
config = Config.from_yaml("configs/medium.yaml")

# Create tokenizer and datasets
tokenizer = Tokenizer(
    tokenizer_type=config.data.tokenizer_type,
    vocab_size=config.data.vocab_size
)
tokenizer.train(your_text_data)

# Create data loaders
train_loader, val_loader = create_data_loaders(config, tokenizer)

# Initialize trainer
trainer = Trainer(model, config, train_loader, val_loader, tokenizer)

# Start training
trainer.train(epochs=100)
```

### Text Generation
```python
# Generate text from trained model
prompt = "The transformer model"
prompt_ids = tokenizer.encode(prompt)
prompt_tensor = torch.tensor([prompt_ids])

generated_ids = model.generate(
    prompt_tensor,
    max_len=100,
    temperature=0.8,
    top_k=10
)

generated_text = tokenizer.decode(generated_ids[0].tolist())
print(f"Generated: {generated_text}")
```

## 📁 Project Structure

```
Hibiscus/
├── transformer/                    # Core transformer implementation
│   ├── __init__.py
│   ├── model.py                   # Main transformer architecture
│   ├── attention.py               # Attention mechanisms
│   ├── positional_encoding.py     # Positional encoding modules
│   ├── config.py                  # Configuration management
│   ├── trainer.py                 # Training infrastructure
│   ├── data.py                    # Data loading and preprocessing
│   ├── tokenizer.py               # Tokenization utilities
│   ├── analysis.py                # Model analysis and visualization
│   └── utils.py                   # Utility functions
├── configs/                       # Configuration files
│   ├── small.yaml                 # Small model (12M parameters)
│   ├── medium.yaml                # Medium model (85M parameters)
│   └── large.yaml                 # Large model (355M parameters)
├── tests/                         # Test suite
│   ├── test_model.py              # Model tests
│   ├── test_attention.py          # Attention mechanism tests
│   └── test_trainer.py            # Training infrastructure tests
├── scripts/                       # Training and evaluation scripts
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation script
│   └── evaluate_model.py          # Comprehensive model evaluation
├── examples/                      # Usage examples
│   └── basic_usage.py             # Complete usage demonstration
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation
└── README.md                      # This file
```

## ⚙️ Configuration

### Model Configurations

The transformer supports three pre-configured model sizes:

#### Small Model (12M parameters)
```yaml
model:
  vocab_size: 30000
  d_model: 512
  n_heads: 8
  n_layers: 6
  d_ff: 2048
  max_seq_len: 512
  dropout: 0.1
```

#### Medium Model (85M parameters)
```yaml
model:
  vocab_size: 30000
  d_model: 768
  n_heads: 12
  n_layers: 12
  d_ff: 3072
  max_seq_len: 512
  dropout: 0.1
```

#### Large Model (355M parameters)
```yaml
model:
  vocab_size: 30000
  d_model: 1024
  n_heads: 16
  n_layers: 24
  d_ff: 4096
  max_seq_len: 512
  dropout: 0.1
```

### Training Configuration
```yaml
training:
  batch_size: 32
  learning_rate: 1e-4
  warmup_steps: 4000
  max_epochs: 100
  gradient_clip_val: 1.0
  optimizer: "adam"
  weight_decay: 0.01
  use_amp: true
  use_ddp: false
```

## 🧪 Testing

### Run All Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=transformer --cov-report=html

# Run specific test file
pytest tests/test_model.py

# Run specific test
pytest tests/test_model.py::test_transformer_forward
```

### Test Coverage
The test suite covers:
- Model architecture and forward pass
- Attention mechanisms and masks
- Training infrastructure and optimizers
- Data loading and tokenization
- Configuration validation
- Integration tests

## 📊 Training

### Basic Training
```bash
# Train with sample data
python scripts/train.py --config configs/small.yaml --create-sample-data

# Train with custom data
python scripts/train.py --config configs/medium.yaml --data-dir /path/to/data

# Resume from checkpoint
python scripts/train.py --config configs/large.yaml --checkpoint checkpoints/model.pt
```

### Advanced Training Options
```bash
# Multi-GPU training
python scripts/train.py --config configs/large.yaml --use-ddp

# Custom number of epochs
python scripts/train.py --config configs/medium.yaml --epochs 50

# Specific device
python scripts/train.py --config configs/small.yaml --device cuda:1
```

## 🔍 Evaluation

### Model Evaluation
```bash
# Comprehensive evaluation
python scripts/evaluate_model.py \
    --checkpoint checkpoints/model.pt \
    --config configs/medium.yaml \
    --test-data data/test \
    --benchmark \
    --generate-samples \
    --analyze-attention
```

### Performance Benchmarking
```bash
# Benchmark inference performance
python scripts/evaluate_model.py \
    --checkpoint checkpoints/model.pt \
    --config configs/large.yaml \
    --benchmark
```

### Attention Visualization
```bash
# Analyze attention patterns
python scripts/evaluate_model.py \
    --checkpoint checkpoints/model.pt \
    --config configs/medium.yaml \
    --analyze-attention
```

## 📈 Monitoring

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir logs/

# View training progress
# Open http://localhost:6006 in your browser
```

### Weights & Biases
```yaml
# Enable W&B in config
use_wandb: true
project_name: "hibiscus_transformer"
experiment_name: "transformer_experiment"
```

## 🛠️ Development

### Code Quality
```bash
# Format code
black transformer/ tests/ scripts/

# Lint code
flake8 transformer/ tests/ scripts/

# Type checking
mypy transformer/

# Run all quality checks
pre-commit run --all-files
```

### Adding New Features
1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement feature with tests
3. Run test suite: `pytest tests/`
4. Format and lint code
5. Submit pull request

## 📚 Examples

### Complete Usage Example
```python
# See examples/basic_usage.py for a complete demonstration
python examples/basic_usage.py
```

This example demonstrates:
- Model creation and configuration
- Tokenizer training and usage
- Training pipeline setup
- Text generation
- Configuration management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite
5. Format and lint your code
6. Submit a pull request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

This implementation is inspired by:
- "Attention Is All You Need" (Vaswani et al., 2017)
- Modern transformer implementations (Hugging Face, PyTorch)
- Best practices from the deep learning community

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation
- Review the examples

---

**This transformer implementation provides a production-ready foundation for natural language processing tasks with comprehensive tooling, extensive testing, and professional development practices.** 