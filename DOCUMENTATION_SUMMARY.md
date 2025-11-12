# Documentation Updates Summary

## Overview

Updated documentation across Keras, PyTorch, and ONNX modules to guide users on the recommended workflow for neural networks with skip connections and residual architectures.

## Key Message

**For models with complex architectures (skip connections, ResNet, etc.):**
- Export to ONNX format first
- Use `add_onnx_dag_constr` for full DAG support
- Direct Keras/PyTorch APIs only support sequential models

## Files Updated

### 1. Keras Documentation (`src/gurobi_ml/keras/keras.py`)

**Added Section:** "Models with Skip Connections or Residual Architectures"

**Content:**
- Explains when to use ONNX export (skip connections, Functional API, etc.)
- Two export methods with code examples:
  - tf2onnx (recommended)
  - keras2onnx (alternative)
- Complete workflow from export to Gurobi ML usage
- Links to external documentation

**Key References:**
- https://github.com/onnx/tensorflow-onnx (tf2onnx)
- https://github.com/onnx/keras-onnx (keras2onnx)
- https://onnx.ai/get-started.html (ONNX)

### 2. PyTorch Documentation (`src/gurobi_ml/torch/sequential.py`)

**Added Section:** "Models with Skip Connections or Residual Architectures"

**Content:**
- Explains limitations of torch.nn.Sequential
- Complete torch.onnx.export example with all parameters
- Model verification steps
- Workflow from export to Gurobi ML usage
- Links to official tutorials

**Key References:**
- https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
- https://pytorch.org/docs/stable/onnx.html
- https://onnx.ai/get-started.html

### 3. ONNX Module Documentation (`src/gurobi_ml/onnx/__init__.py`)

**Enhanced:** Module-level docstring

**Content:**
- Clear distinction between add_onnx_constr and add_onnx_dag_constr
- Quick reference examples for Keras and PyTorch export
- Recommended workflow overview

### 4. Comprehensive Guide (`ONNX_EXPORT_GUIDE.md`)

**New File:** Complete tutorial and reference

**Sections:**
1. Overview - When to use ONNX export
2. Keras Models with Skip Connections
   - Full example with skip connection
   - Step-by-step export process
   - Gurobi ML integration
3. PyTorch Models with Residual Connections
   - ResNet-style example
   - Detailed export parameters
   - Accuracy verification
4. Supported ONNX Operations
5. Accuracy Information
6. Installation Requirements
7. Common Issues and Solutions
8. References

## Documentation Style

All documentation follows these principles:

1. **Problem First:** Explains when users need ONNX export
2. **Step-by-Step:** Clear numbered steps with code examples
3. **Complete Code:** Runnable examples, not fragments
4. **External Links:** References to official documentation
5. **Troubleshooting:** Common issues and solutions

## Example Workflow Documented

### Keras
```python
# 1. Export to ONNX
import tf2onnx
spec = (tf.TensorSpec((None, input_dim), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(keras_model, input_signature=spec)
onnx.save(model_proto, "model.onnx")

# 2. Use with Gurobi ML
from gurobi_ml.onnx import add_onnx_dag_constr
onnx_model = onnx.load("model.onnx")
pred = add_onnx_dag_constr(gp_model, onnx_model, input_vars)
```

### PyTorch
```python
# 1. Export to ONNX
torch.onnx.export(pytorch_model, dummy_input, "model.onnx",
                  export_params=True, opset_version=11)

# 2. Use with Gurobi ML
from gurobi_ml.onnx import add_onnx_dag_constr
onnx_model = onnx.load("model.onnx")
pred = add_onnx_dag_constr(gp_model, onnx_model, input_vars)
```

## Impact

**Before:**
- Users with ResNet/skip connection models would get NoModel errors
- No guidance on alternatives
- Frustration and confusion

**After:**
- Clear explanation of limitations
- Step-by-step ONNX export workflow
- Links to official documentation
- Working examples
- Troubleshooting guide

## Testing

All documentation has been verified for:
- ✓ Python syntax correctness
- ✓ Presence of key sections (Skip Connections, ONNX export)
- ✓ References to export functions (tf2onnx, torch.onnx.export)
- ✓ Links to external documentation
- ✓ Complete code examples

## Related Files

- `DAG_IMPLEMENTATION_SUMMARY.md` - Technical implementation details
- `COMPLETE_SOLUTION_SUMMARY.md` - Complete solution overview
- `DEMO_DAG_IMPLEMENTATION.py` - Working demonstration script

## Future Considerations

If direct Keras/PyTorch DAG support is added later:
1. Update these documentation sections
2. Add new API functions
3. Keep ONNX workflow as alternative
4. Maintain backward compatibility
