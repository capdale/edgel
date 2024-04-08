# Edgel
Convert network to edge device compatible, easy to test, benchmark performance from any model, pytorch, huggingface, tensorflow  

# How to use
```python
import tensorflow as tf
from edgel import tf as etf

a = etf.load('tf-model-path')
# add more supported ops
a.export_as_tflite('export.tflite', supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS])
```

# List of model
|Framework|version|
|---|---|
|Tensorflow (tf concrete only) |2.15.0|
|Pytorch| x |
|Onnx| x |
|JAX | x |
| huggingface (optimum) | x |