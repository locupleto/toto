# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
This repository contains Toto (Time Series Optimized Transformer for Observability), a foundation model for multivariate time series forecasting. The project also includes BOOM (Benchmark of Observability Metrics), a large-scale forecasting dataset.

## Development Setup

### Installation Commands

#### Basic Installation (Required)
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in editable mode (includes all required dependencies)
pip install --editable .
```

#### Optional Performance Optimizations
```bash
# For optimal performance, install additional dependencies (OPTIONAL)
# Note: xformers may have installation issues on some systems (especially macOS)
# The project works fully without these - they only provide performance improvements
pip install xformers flash-attention
```

**Important:** xformers and flash-attention are **optional** dependencies. The project includes automatic fallbacks to native PyTorch implementations if these libraries are not available. All core functionality works without them, though performance may be slightly slower.

### Development Commands
```bash
# Run tests
pytest                              # Run all tests
pytest toto/test/                   # Run tests in specific directory
pytest -m cuda                     # Run CUDA tests only
pytest -m "not cuda"               # Skip CUDA tests

# Code formatting and linting
black .                             # Format code
isort .                             # Sort imports
mypy toto/                          # Type checking

# Run LSF evaluation
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$(pwd):$(pwd)/toto:$PYTHONPATH"
python toto/evaluation/run_lsf_eval.py --datasets ETTh1 --context-length 2048 --eval-stride 1 --checkpoint-path [CHECKPOINT-NAME-OR-DIR]
```

## Architecture

### Core Components

#### Model Architecture (`toto/model/`)
- **`toto.py`**: Main model class with HuggingFace Hub integration
- **`backbone.py`**: TotoBackbone - core transformer architecture with patch embedding
- **`transformer.py`**: Transformer implementation with alternating time/space attention
- **`attention.py`**: Proportional Factorized Space-Time Attention mechanism
- **`embedding.py`**: PatchEmbedding for converting time series to patches
- **`scaler.py`**: Various scaling strategies for time series normalization
- **`distribution.py`**: Output distribution classes (Student-T mixture)

#### Inference (`toto/inference/`)
- **`forecaster.py`**: TotoForecaster - autoregressive forecasting with KV cache
- **`gluonts_predictor.py`**: GluonTS-compatible predictor interface

#### Data (`toto/data/`)
- **`dataset.py`**: MaskedTimeseries class and utility functions for data handling

#### Evaluation (`toto/evaluation/`)
- **`run_lsf_eval.py`**: LSF benchmark evaluation script
- **`lsf/`**: LSF datasets and evaluation utilities
- **`gift_eval/`**: GIFT-Eval benchmark integration

### Key Design Patterns

1. **Patch-based Processing**: Time series are divided into patches (like tokens in NLP)
2. **Alternating Attention**: Transformer alternates between time-wise and space-wise attention
3. **Autoregressive Forecasting**: Future predictions are generated step-by-step
4. **Probabilistic Outputs**: Model outputs distributions, not just point estimates
5. **Multi-scale Architecture**: Supports various context lengths and prediction horizons

### Data Flow
1. Input time series → PatchEmbedding → Transformer → Distribution parameters
2. For inference: Historical data → Autoregressive decoding → Future predictions
3. Scaling applied before/after model processing for numerical stability

## Testing

### Test Structure and Organization
```
toto/test/
├── helper_functions.py          # Test utilities and xformers detection
├── model/
│   ├── scaler_test.py          # ✅ Causal scaling algorithms (6 tests)
│   └── attention_test.py       # ⚠️ Attention mechanisms (requires xformers)
└── inference/
    ├── forecaster_test.py      # ⚠️ TotoForecaster tests (requires xformers)
    └── gluonts_predictor_test.py # ⚠️ GluonTS integration (requires xformers)
```

### Test Categories and Markers
- **`@pytest.mark.cuda`**: GPU tests (require CUDA device)
- **`@pytest.mark.stress`**: Stress tests for performance validation
- **`@pytest.mark.real_aws`**: Tests requiring AWS credentials
- **xformers-dependent**: Automatically skipped if xformers unavailable

### Running Tests

#### Basic Test Commands
```bash
# Run all available tests (recommended)
pytest                              # 6 pass, 3 skip (without xformers)
pytest -v                          # Verbose output with test names
pytest -m "not cuda"               # Skip CUDA tests (for CPU-only)
pytest toto/test/model/             # Run only model tests (all pass)
pytest --tb=short                  # Shorter traceback format
```

#### Test Status Overview
- ✅ **Model Tests (6/6 passing)**: Core mathematical algorithms
  - `CausalPatchStdMeanScaler` with/without Bessel correction
  - `CausalStdMeanScaler` with padding and weights
  - Edge cases and numerical stability
- ⚠️ **Inference Tests (3 skipped)**: Require xformers but have fallbacks
  - Mock-based forecaster tests with comprehensive scenarios
  - GluonTS predictor integration tests

#### Test Environment Configuration
- **pytest.ini**: Configures markers, warnings, and test paths
- **Environment variables**: `PYTORCH_ENABLE_MPS_FALLBACK=1` for macOS MPS
- **Auto-skip logic**: `skip_if_no_xformers()` gracefully handles missing dependencies
- **Device detection**: `set_default_dtype()` optimizes for available hardware

### Test Development Guidelines
- All tests use `beartype` for runtime type checking
- Mock objects simulate model behavior for unit testing
- Comprehensive edge case coverage (padding, weights, different shapes)
- Numerical precision testing with `torch.testing.assert_close()`

## Configuration

### Python Environment
- Target Python version: 3.10+
- PyTorch 2.5+ required
- CUDA Ampere generation or newer recommended

### Code Quality
- Black for code formatting (line length: 120)
- isort for import sorting (black profile)
- mypy for type checking
- Excludes: `toto/lotsa_data/` directory from mypy

### Dependencies
Key dependencies include:
- torch==2.7.0
- transformers==4.50.0
- gluonts[torch]==0.15.1
- einops==0.7.0
- jaxtyping==0.2.29
- beartype==0.18.5

## Common Workflows

### Complete Inference Tutorial

The main inference tutorial is available at `toto/notebooks/inference_tutorial.ipynb`. Here's the complete workflow:

#### 1. Data Preparation
```python
import torch
from toto.data.util.dataset import MaskedTimeseries

# Prepare time series data (Variate × Time Steps format)
# Example: ETTm1 dataset with 7 variates, 4096 context steps
feature_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
input_series = torch.from_numpy(df[feature_columns].values.T).to(torch.float).to(device)
# Shape: [7, 4096]

# Create timestamp features (required by API, not currently used by model)
timestamp_seconds = torch.from_numpy(df.timestamp_seconds.values.T).expand((n_variates, context_length))
time_interval_seconds = torch.full((n_variates,), 60*15)  # 15-min intervals

# Create MaskedTimeseries input
inputs = MaskedTimeseries(
    series=input_series,
    padding_mask=torch.full_like(input_series, True, dtype=torch.bool),  # 1=valid, 0=padding
    id_mask=torch.zeros_like(input_series),  # For packing different series
    timestamp_seconds=timestamp_seconds,
    time_interval_seconds=time_interval_seconds,
)
```

#### 2. Model Loading and Setup
```python
from toto.model.toto import Toto
from toto.inference.forecaster import TotoForecaster

# Load pretrained model from HuggingFace Hub
toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
toto.to(device)

# Optional: Enable JIT compilation for repeated inference
toto.compile()

# Create forecaster
forecaster = TotoForecaster(toto.model)
```

#### 3. Generate Forecasts
```python
forecast = forecaster.forecast(
    inputs,
    prediction_length=336,        # Number of steps to forecast
    num_samples=256,             # Monte Carlo samples for uncertainty
    samples_per_batch=256,       # Batch size for memory management
    use_kv_cache=True,          # Enable KV cache for speed
)

# Access results
point_forecast = forecast.mean           # Shape: [batch, variates, prediction_length]
samples = forecast.samples              # Shape: [batch, variates, prediction_length, num_samples]
quantiles = samples.quantile(q=torch.tensor([0.05, 0.95]), dim=-1)  # 90% prediction intervals
```

#### 4. Key Parameters and Considerations
- **Context Length**: 4096 max (training length), but can extrapolate longer
- **Prediction Length**: Any length (autoregressive generation)
- **Samples**: 256 recommended for stable uncertainty estimates
- **Device Support**: CUDA preferred, CPU and MPS supported
- **Memory**: Use `samples_per_batch` to control memory usage

### Evaluation Workflows
- **LSF evaluation**: Use `run_lsf_eval.py` with specific datasets
- **GIFT-Eval**: Use provided notebook in `toto/evaluation/gift_eval/`
- **BOOM evaluation**: Use notebook in `boom/notebooks/`
- **Inference Tutorial**: `toto/notebooks/inference_tutorial.ipynb` - complete ETTm1 example

## Performance Considerations
- Use model compilation: `toto.compile()` for faster inference
- KV cache enabled by default for autoregressive generation
- Batch processing for multiple samples to optimize memory usage
- xformers and flash-attention provide performance improvements but are optional
- The project includes automatic fallbacks:
  - RMSNorm: xformers fused kernel → native PyTorch implementation
  - SwiGLU: xformers fused kernel → native PyTorch implementation  
  - Attention: memory-efficient attention → scaled_dot_product_attention

## Troubleshooting and Common Issues

### Installation Issues
**Problem**: xformers installation fails (especially on macOS)
```bash
# Solution: Skip xformers - project works without it
pip install --editable .  # Only install required dependencies
# xformers tests will auto-skip, fallbacks are used
```

**Problem**: Module import errors after installation
```bash
# Solution: Ensure editable install and check Python path
pip install --editable .
python -c "import toto; print('Success')"
```

### Development Workflow Issues
**Problem**: Tests failing or being skipped
```bash
# Check test status
pytest --collect-only    # See what tests are available
pytest -v               # Run with verbose output
pytest toto/test/model/  # Run only working model tests
```

**Problem**: Type checking errors
```bash
# Run type checking (excludes toto/lotsa_data/)
mypy toto/
# Fix imports and type annotations as needed
```

### Runtime Issues
**Problem**: CUDA out of memory during inference
```python
# Solution: Reduce batch size and use CPU fallback
forecast = forecaster.forecast(
    inputs,
    prediction_length=336,
    num_samples=64,          # Reduce from 256
    samples_per_batch=32,    # Reduce batch size
)
# Or move to CPU: toto.to('cpu')
```

**Problem**: Slow inference performance
```python
# Solution: Enable optimizations
toto.compile()              # JIT compilation
use_kv_cache=True          # Enable in forecast()
# Use GPU if available: toto.to('cuda')
```

### Data Format Issues
**Problem**: Wrong tensor shapes or data format
```python
# Correct format: [batch, variates, time_steps] or [variates, time_steps]
input_series = torch.from_numpy(df.values.T)  # Transpose to variate×time
# Not: torch.from_numpy(df.values)  # time×variate (wrong)

# Ensure float dtype
input_series = input_series.to(torch.float)

# Create proper MaskedTimeseries
inputs = MaskedTimeseries(
    series=input_series,
    padding_mask=torch.ones_like(input_series, dtype=torch.bool),
    id_mask=torch.zeros_like(input_series),
    timestamp_seconds=timestamps,
    time_interval_seconds=intervals,
)
```

### Quick Validation
```python
# Test core functionality without model download
from toto.data.util.dataset import MaskedTimeseries
import torch

# Create dummy data
series = torch.randn(2, 3, 100)  # 2 batch, 3 variates, 100 timesteps
mask = torch.ones_like(series, dtype=torch.bool)
inputs = MaskedTimeseries(
    series=series, 
    padding_mask=mask,
    id_mask=torch.zeros_like(series),
    timestamp_seconds=torch.arange(100).expand(2, 3, 100),
    time_interval_seconds=torch.full((2, 3), 60)
)
print("✓ Core data structures working")
```

## Contributing
Follow RFC process for new features (see CONTRIBUTING.md). Bug fixes can be submitted directly as PRs. All contributions require:
- Code formatting with black/isort
- Type checking with mypy
- Appropriate tests
- Documentation updates if needed