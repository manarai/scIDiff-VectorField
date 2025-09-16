# scIDiff - Corrected Version

This is a corrected version of the scIDiff repository that fixes multiple installation and functionality issues found in the original repository.

## Issues Fixed

### 1. Package Structure Issues
- **Fixed**: Wrong init filename `_init_.py` → `__init__.py`
- **Fixed**: Empty package init file - now properly imports main components
- **Added**: Complete CLI modules (train.py, sample.py, design.py) referenced in setup.py

### 2. Dependency Issues
- **Fixed**: Python 3.11 compatibility by updating numba>=0.57.0 and llvmlite>=0.40.0
- **Fixed**: Duplicate dependencies in requirements.txt (biomart, mygene were listed twice)
- **Added**: Missing dynamo and optimal transport dependencies:
  - dynamo-release>=1.3.0
  - pot>=0.8.0
  - ott-jax>=0.4.0
  - jax>=0.4.0
  - jaxlib>=0.4.0

### 3. Enhanced Functionality
- **Added**: Complete Schrödinger Bridge implementation (`scIDiff/bridge.py`)
- **Added**: Comprehensive Dynamo integration (`scIDiff/dynamo_integration.py`)
- **Added**: Vector field velocyto integration
- **Added**: Full optimal transport capabilities

## New Features

### Schrödinger Bridge (`scIDiff.bridge`)
```python
from scIDiff.bridge import SchrodingerBridge, DynamoBridge

# Create bridge for perturbation modeling
bridge = SchrodingerBridge(gene_dim=2000)

# Train bridge between control and perturbed states
loss = bridge.bridge_loss(control_cells, perturbed_cells)

# Sample bridge trajectory
trajectory = bridge.sample_bridge(control_cells, perturbed_cells)

# Predict perturbation effects
predict_fn = bridge.predict_perturbation(control_cells, perturbed_cells)
predicted = predict_fn(new_control_cells)
```

### Dynamo Integration (`scIDiff.dynamo_integration`)
```python
from scIDiff.dynamo_integration import DynamoVectorField, create_dynamo_guided_model

# Fit vector field to data
vf = DynamoVectorField(adata)
vf.fit_vector_field()

# Create guided diffusion model
guided_model = create_dynamo_guided_model(gene_dim=2000, adata=adata)

# Sample with vector field guidance
samples = guided_model.sample_with_guidance(x_init)
```

### Command Line Interface
```bash
# Train model
scidiff-train --data data.h5ad --output model.pt --epochs 100

# Generate samples
scidiff-sample --model model.pt --num-samples 1000 --gene-dim 2000

# Inverse design
scidiff-design --model model.pt --targets targets.json --gene-dim 2000
```

## Installation

### Option 1: Install from corrected source
```bash
cd scIDiff_corrected
pip install -e .
```

### Option 2: Install core dependencies first (recommended)
```bash
# Install core dependencies
pip install torch torchvision numpy scipy pandas matplotlib

# Install biological dependencies
pip install scanpy anndata

# Install optional dependencies for full functionality
pip install dynamo-release pot jax jaxlib

# Then install scIDiff
cd scIDiff_corrected
pip install -e .
```

## Testing

The corrected version has been tested and verified to work:

```python
import scIDiff
from scIDiff import ScIDiffModel, ScIDiffTrainer, InverseDesigner

# Create and test model
model = ScIDiffModel(gene_dim=100)
samples = model.sample(batch_size=4)
print(f"Generated samples shape: {samples.shape}")

# Test bridge functionality
from scIDiff.bridge import SchrodingerBridge
bridge = SchrodingerBridge(gene_dim=50)

# Test dynamo integration (with warnings for missing optional deps)
from scIDiff.dynamo_integration import VectorFieldSurrogate
surrogate = VectorFieldSurrogate(input_dim=50)
```

## Key Improvements

1. **Python 3.11 Compatibility**: Updated all dependencies to work with modern Python
2. **Complete Package Structure**: All referenced modules now exist and work
3. **Enhanced OT Implementation**: Full Schrödinger bridge with entropic optimal transport
4. **Dynamo Integration**: Complete vector field integration for RNA velocity analysis
5. **Working CLI**: Functional command-line tools for training, sampling, and design
6. **Better Documentation**: Clear usage examples and installation instructions

## Dependencies

### Core (Required)
- torch>=1.12.0
- numpy>=1.21.0
- scipy>=1.7.0
- pandas>=1.3.0

### Biological (Recommended)
- scanpy>=1.9.0
- anndata>=0.8.0

### Advanced Features (Optional)
- dynamo-release>=1.3.0 (for vector field analysis)
- pot>=0.8.0 (for optimal transport)
- jax>=0.4.0 (for advanced OT algorithms)
- scvelo (for RNA velocity analysis)

## Version History

- **v0.2.0**: Corrected version with all fixes and enhancements
- **v0.1.0**: Original version with installation issues

## Notes

- The dynamo integration will show warnings if dynamo-release or scvelo are not installed, but core functionality works without them
- CLI modules require scanpy for data loading
- Full optimal transport features require pot and jax packages
- All core diffusion model functionality works with just torch and numpy

