# Using DAG Neural Networks with Gurobi ML

This guide explains how to use neural networks with skip connections, residual architectures, and other non-sequential topologies in Gurobi ML.

## Overview

Gurobi ML supports two types of neural network architectures:

1. **Sequential Models**: Linear chains of layers (Dense → ReLU → Dense → ...)
   - Supported directly via `add_keras_constr` and `add_sequential_constr`
   
2. **DAG Models**: Arbitrary directed acyclic graphs with skip/residual connections
   - Supported via ONNX export + `add_onnx_dag_constr`

## When to Use ONNX Export

Use the ONNX export workflow if your model has:

- ✓ Skip connections (input used by multiple layers)
- ✓ Residual connections (ResNet-style architectures)
- ✓ Multi-branch architectures
- ✓ Keras Functional API models
- ✓ Custom PyTorch modules with complex forward() logic

## Keras Models with Skip Connections

### Step 1: Create Your Keras Model

```python
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Add, ReLU

# Example: Model with skip connection
inputs = Input(shape=(4,))

# Main branch
x1 = Dense(8)(inputs)
x1 = ReLU()(x1)
x1 = Dense(2)(x1)

# Skip connection branch
x2 = Dense(2)(inputs)  # Direct path from input to output

# Combine branches
outputs = Add()([x1, x2])

model = keras.Model(inputs=inputs, outputs=outputs)
```

### Step 2: Export to ONNX

```python
import tf2onnx
import onnx

# Convert Keras model to ONNX
spec = (tf.TensorSpec((None, 4), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)

# Save to file
onnx.save(model_proto, "keras_skip_model.onnx")

# Verify the exported model
onnx_model = onnx.load("keras_skip_model.onnx")
onnx.checker.check_model(onnx_model)
```

### Step 3: Use with Gurobi ML

```python
import gurobipy as gp
from gurobi_ml.onnx import add_onnx_dag_constr
import onnx

# Load ONNX model
onnx_model = onnx.load("keras_skip_model.onnx")

# Create Gurobi model
gp_model = gp.Model()
x = gp_model.addMVar(shape=(4,), lb=-10, ub=10, name="x")

# Add predictor constraint with DAG support
pred = add_onnx_dag_constr(gp_model, onnx_model, x)

# Use the model
gp_model.setObjective(pred.output.sum(), gp.GRB.MINIMIZE)
gp_model.optimize()

print(f"Optimal input: {x.X}")
print(f"Predicted output: {pred.output.X}")
```

## PyTorch Models with Residual Connections

### Step 1: Create Your PyTorch Model

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim, dim)
    
    def forward(self, x):
        identity = x  # Skip connection
        
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        
        out = out + identity  # Residual addition
        return out

class ResNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(4, 8)
        self.residual_block = ResidualBlock(8)
        self.output_layer = nn.Linear(8, 2)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_block(x)  # Residual connection inside
        x = self.output_layer(x)
        return x

model = ResNetModel()
model.eval()
```

### Step 2: Export to ONNX

```python
import torch
import onnx

# Prepare dummy input with correct shape
dummy_input = torch.randn(1, 4)

# Export to ONNX
torch.onnx.export(
    model,                      # Your model
    dummy_input,                # Example input
    "pytorch_resnet.onnx",     # Output file
    export_params=True,         # Store trained weights
    opset_version=11,           # ONNX version
    do_constant_folding=True,   # Optimize constant operations
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# Verify the exported model
onnx_model = onnx.load("pytorch_resnet.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX export successful!")
```

### Step 3: Use with Gurobi ML

```python
import gurobipy as gp
from gurobi_ml.onnx import add_onnx_dag_constr
import onnx
import numpy as np

# Load ONNX model
onnx_model = onnx.load("pytorch_resnet.onnx")

# Create Gurobi model
gp_model = gp.Model()
x = gp_model.addMVar(shape=(4,), lb=-5, ub=5, name="x")

# Add predictor constraint with DAG support
pred = add_onnx_dag_constr(gp_model, onnx_model, x)

# Verify accuracy by comparing with PyTorch
test_input = np.random.randn(1, 4).astype(np.float32)

# Set input in Gurobi
for i in range(4):
    x[i].lb = x[i].ub = test_input[0, i]

gp_model.optimize()

# Compare outputs
pytorch_output = model(torch.from_numpy(test_input)).detach().numpy()
gurobi_output = pred.output.X

print(f"PyTorch output: {pytorch_output[0]}")
print(f"Gurobi output:  {gurobi_output}")
print(f"Max error:      {np.abs(pytorch_output[0] - gurobi_output).max():.2e}")
```

## Supported ONNX Operations

The DAG implementation supports:

- **Gemm**: Fully connected layers (with transB support)
- **MatMul**: Matrix multiplication
- **Add**: Both bias addition and residual connections
- **Relu**: ReLU activation
- **Identity**: Pass-through nodes

## Accuracy

The DAG implementation maintains high accuracy:
- Typical error: < 1e-7 compared to ONNX Runtime
- Tested with real-world models (ResNet-style architectures)
- Identical optimization performance to sequential models

## Installation Requirements

For ONNX export, you'll need:

**Keras/TensorFlow:**
```bash
pip install tf2onnx onnx
# or
pip install keras2onnx onnx
```

**PyTorch:**
```bash
pip install torch onnx
```

**Gurobi ML:**
```bash
pip install gurobipy gurobi-machinelearning
```

## References

- [ONNX Official Documentation](https://onnx.ai/)
- [PyTorch ONNX Export Tutorial](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
- [tf2onnx GitHub Repository](https://github.com/onnx/tensorflow-onnx)
- [Gurobi ML Documentation](https://gurobi-machinelearning.readthedocs.io/)

## Common Issues and Solutions

### Issue: ONNX export fails with dynamic shapes

**Solution:** Specify dynamic_axes in torch.onnx.export:
```python
torch.onnx.export(
    model, dummy_input, "model.onnx",
    dynamic_axes={'input': {0: 'batch_size'}}
)
```

### Issue: Unsupported ONNX operation

**Solution:** Check if the operation is in the supported list. If not, consider:
1. Simplifying the model architecture
2. Replacing unsupported operations with supported alternatives
3. Opening an issue on Gurobi ML GitHub

### Issue: Accuracy mismatch between frameworks

**Solution:** Verify ONNX export first:
```python
import onnxruntime as ort

# Test ONNX model
ort_session = ort.InferenceSession("model.onnx")
onnx_output = ort_session.run(None, {'input': test_input})[0]

# Compare with original framework
original_output = model.predict(test_input)  # or model(test_input)
print(f"Error: {np.abs(onnx_output - original_output).max()}")
```

## Complete Example

See `DEMO_DAG_IMPLEMENTATION.py` in the repository for a complete working example.
