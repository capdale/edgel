from __future__ import annotations

from ..base.exporter import Exporter

from typing import Any, List
import tensorflow as tf

class TFWrapper(Exporter):
    def __init__(self) -> None:
        pass
    
    def export_as_tflite(self):
        raise NotImplementedError()

class TFConcrete(TFWrapper):
    def __init__(self, loaded_model) -> None:
        self.loaded_model = loaded_model

        # default signature
        self.cur_sig_name = 'serving_default'
        self.signature_keys = self.loaded_model.signatures.keys()

        if len(self.signature_keys) == 0:
            raise ValueError("wrong tf concrete model")

        if not 'serving_default' in self.signature_keys:
            self.cur_sig_name = self.signature_keys[0]

        self.cur_sig = self.loaded_model.signatures[self.cur_sig_name]

    @property
    def cur_signature(self) -> str:
        return self.cur_sig_name

    @property
    def signatures(self) -> List[str]:
        return self.signature_keys
    
    def change_signature(self, signature: str):
        if not signature in self.signature_keys:
            raise ValueError(f"{signature} is not exist")
        self.cur_sig_name = signature
        self.cur_sig = self.loaded_model.signatures[self.cur_sig_name]
    
    def input_spec(self) -> tf.TensorSpec:
        return self.cur_sig.structured_input_signature
    
    def output_spec(self) -> tf.TensorSpec:
        return self.cur_sig.structured_outputs
    
    def __call__(self, **x):
        return self.cur_sig(**x)

    def export_as_tflite(self, path: str, signature_names: List[str] | None=None, experimental_new_converter: bool | None=None, supported_ops: List[tf.lite.OpsSet] | None=None):
        signatures = []
        if signature_names is None:
            signature_names = [self.cur_sig_name]
        for sig in signature_names:
            if not sig in self.signature_keys:
                raise ValueError(f"{sig} is not exist")
        for sig in signature_names:
            signatures.append(self.loaded_model.signatures[sig])

        converter = tf.lite.TFLiteConverter.from_concrete_functions(signatures, self.loaded_model)
        if experimental_new_converter is not None:
            converter.experimental_new_converter = experimental_new_converter
        if supported_ops is not None:
            converter.target_spec.supported_ops = supported_ops
        
        litemodel = converter.convert()
        with open(path, "wb") as f:
            f.write(litemodel)

    @classmethod
    def load_with_saved_model(cls, path: str) -> TFConcrete:
        loaded = tf.saved_model.load(path)
        return TFConcrete(loaded) 

    
