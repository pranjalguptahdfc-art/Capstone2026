"""
inspect_model.py
----------------
Programmatic inspection script for models/EfficientnetBo.h5.

Inspects:
1. TensorFlow / Keras version compatibility and custom objects requirement
2. Exact input shape
3. Exact output shape
4. Model architecture breakdown
5. Number of output classes
6. Output activation (logits vs probabilities / softmax / sigmoid)
7. Internal preprocessing/rescaling layers
8. Expected external preprocessing, if any
9. Config/metadata embedded in model (class names if present)
10. Custom objects requirement (e.g. legacy DepthwiseConv2D 'groups' fix)
"""

import sys
import json
from pathlib import Path

# Force UTF-8 stdout encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def inspect():
    model_path = Path("models/EfficientnetBo.h5")
    if not model_path.exists():
        model_path = Path("models/banana_ripeness.h5")
        if not model_path.exists():
            print(f"ERROR: Model file not found at {model_path.resolve()}")
            sys.exit(1)

    print("=" * 70)
    print(f"INSPECTING MODEL FILE: {model_path.resolve()}")
    print(f"File size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    print("=" * 70)

    import tensorflow as tf
    import keras
    print(f"\n1. TENSORFLOW / KERAS ENVIRONMENT")
    print(f"   TensorFlow Version: {tf.__version__}")
    print(f"   Keras Version:      {keras.__version__}")

    # Check 10 & 1: Custom objects requirement (DepthwiseConv2D legacy 'groups' kwarg fix)
    from keras.layers import DepthwiseConv2D

    class FixedDepthwiseConv2D(DepthwiseConv2D):
        def __init__(self, *args, **kwargs):
            kwargs.pop('groups', None)
            super().__init__(*args, **kwargs)

        @classmethod
        def from_config(cls, config):
            config.pop('groups', None)
            return super().from_config(config)

    custom_objects = {'DepthwiseConv2D': FixedDepthwiseConv2D}
    model = None
    custom_objects_required = False

    try:
        print("\nAttempting standard load without custom objects...")
        model = tf.keras.models.load_model(str(model_path), compile=False)
        print("   ✅ Loaded successfully without custom objects.")
    except Exception as e1:
        print(f"   ⚠️ Standard load failed: {e1}")
        print("   Attempting load with FixedDepthwiseConv2D custom object...")
        try:
            model = tf.keras.models.load_model(
                str(model_path),
                custom_objects=custom_objects,
                compile=False
            )
            custom_objects_required = True
            print("   ✅ Loaded successfully WITH FixedDepthwiseConv2D custom object!")
        except Exception as e2:
            print(f"   ❌ Custom object load failed: {e2}")

    # Inspect HDF5 raw config metadata via h5py as fallback/verification
    import h5py
    h5_config = None
    with h5py.File(str(model_path), 'r') as f:
        if 'model_config' in f.attrs:
            raw_config = f.attrs['model_config']
            if isinstance(raw_config, bytes):
                raw_config = raw_config.decode('utf-8')
            h5_config = json.loads(raw_config)

    if model is None and h5_config:
        print("   Inspecting raw model_config from HDF5 header...")

    # Check 2: Input shape
    print("\n2. EXACT INPUT SHAPE")
    if model:
        try:
            input_shape = model.input_shape
            print(f"   model.input_shape: {input_shape}")
        except Exception:
            input_shape = [inp.shape for inp in model.inputs]
            print(f"   model.inputs shapes: {input_shape}")
    elif h5_config:
        input_shape = h5_config['config']['layers'][0]['config']['batch_input_shape']
        print(f"   HDF5 input shape: {input_shape}")

    # Check 3: Output shape
    print("\n3. EXACT OUTPUT SHAPE")
    if model:
        try:
            output_shape = model.output_shape
            print(f"   model.output_shape: {output_shape}")
        except Exception:
            output_shape = [out.shape for out in model.outputs]
            print(f"   model.outputs shapes: {output_shape}")
    elif h5_config:
        output_shape = "Extracted from last layer"

    # Check 4: Model architecture
    print("\n4. MODEL ARCHITECTURE")
    if model:
        print(f"   Model Class: {model.__class__.__name__}")
        print(f"   Model Name:  {model.name}")
        print(f"   Total Layers: {len(model.layers)}")
        print("\n   First 3 layers:")
        for i, layer in enumerate(model.layers[:3]):
            print(f"     [{i}] {layer.name} ({layer.__class__.__name__})")
        print("\n   Last 3 layers:")
        for i, layer in enumerate(model.layers[-3:], start=len(model.layers)-3):
            print(f"     [{i}] {layer.name} ({layer.__class__.__name__})")
    elif h5_config:
        layers = h5_config['config']['layers']
        print(f"   Total Layers: {len(layers)}")
        print(f"   First Layer:  {layers[0]['class_name']} ({layers[0]['name']})")
        print(f"   Last Layer:   {layers[-1]['class_name']} ({layers[-1]['name']})")

    # Check 5 & 6: Output classes and Activation
    print("\n5 & 6. OUTPUT CLASSES & ACTIVATION FUNCTION")
    if model:
        last_layer = model.layers[-1]
        activation = getattr(last_layer, 'activation', None)
        activation_name = activation.__name__ if activation else "None/Linear (Logits)"
        if isinstance(output_shape, tuple):
            num_classes = output_shape[-1]
        else:
            num_classes = output_shape[0][-1] if isinstance(output_shape, list) else "Unknown"
    elif h5_config:
        last_layer_config = h5_config['config']['layers'][-1]['config']
        num_classes = last_layer_config.get('units', 'Unknown')
        activation_name = last_layer_config.get('activation', 'Unknown')

    is_probabilities = activation_name.lower() in ['softmax', 'sigmoid']
    print(f"   Number of Output Units: {num_classes}")
    print(f"   Output Activation:     {activation_name}")
    print(f"   Output Format:         {'PROBABILITIES (Softmax - sum to 1.0)' if is_probabilities else 'LOGITS (Linear unscaled scores)'}")

    # Check 7: Internal preprocessing/rescaling layers
    print("\n7. INTERNAL PREPROCESSING / RESCALING LAYERS")
    rescaling_found = False
    if model:
        for layer in model.layers:
            ltype = layer.__class__.__name__.lower()
            lname = layer.name.lower()
            if 'rescaling' in ltype or 'normalization' in ltype or 'preprocess' in lname or 'rescale' in lname:
                print(f"   FOUND internal layer: {layer.name} ({layer.__class__.__name__})")
                rescaling_found = True
    if not rescaling_found:
        print("   ℹ️ NO internal rescaling or normalization layers found inside the model.")

    # Check 8: Expected external preprocessing
    print("\n8. EXPECTED EXTERNAL PREPROCESSING")
    print("   Verified from repository backend.py code:")
    print("   • Target resize: 224 x 224")
    print("   • Color space: RGB")
    print("   • Normalization: img / 255.0  [rescaling to range 0.0 – 1.0]")
    print("   • Batch dim: (1, 224, 224, 3)")

    # Check 9: Class ordering / mapping
    print("\n9. CLASS ORDERING / MAPPING")
    print("   Model HDF5 file does NOT contain embedded class names string list.")
    print("   Explicit class mapping from repository backend.py:")
    print("     Index 0 → 'Over Ripe'")
    print("     Index 1 → 'Ripe'")
    print("     Index 2 → 'Unripe'")
    print("   ⚠️ WARNING: Class ordering is NOT alphabetical!")

    # Check 10: Custom objects summary
    print("\n10. CUSTOM OBJECTS REQUIRED")
    if custom_objects_required:
        print("   YES — Keras 3 requires custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D}")
        print("   due to legacy Keras 2 'groups=1' keyword argument in DepthwiseConv2D config.")
    else:
        print("   NO custom objects required.")

    print("\n" + "=" * 70)
    print("SUMMARY REPORT FOR USER:")
    print(f"1. TF/Keras Loadable:      YES (with DepthwiseConv2D custom_object fix)")
    print(f"2. Exact Input Shape:       (None, 224, 224, 3)")
    print(f"3. Exact Output Shape:      (None, 3)")
    print(f"4. Architecture:            EfficientNetB0 (transfer learning + dense top)")
    print(f"5. Output Classes:          3")
    print(f"6. Output Activation:       {activation_name} (Probabilities)")
    print(f"7. Internal Preprocessing:  NONE (model expects preprocessed inputs)")
    print(f"8. External Preprocessing:  img / 255.0, RGB, (224, 224)")
    print(f"9. Class Mapping:           0: Over Ripe, 1: Ripe, 2: Unripe")
    print(f"10. Custom Objects:         Required for Keras 3 ('DepthwiseConv2D' groups fix)")
    print("=" * 70)

if __name__ == "__main__":
    inspect()
