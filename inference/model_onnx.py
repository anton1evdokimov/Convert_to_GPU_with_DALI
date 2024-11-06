from pathlib import Path
from time import perf_counter
from typing import Optional, List, Dict, Tuple

import numpy as np
from openvino.runtime import Core

class ModelONNX():
    
    def __init__(self, model_onnx_path: bytes, num_classes: int, desired_classes: Optional[List[int]] = None) -> None:
        self._init_model(model_onnx_path)
        self.num_classes = num_classes
        self.desired_classes = np.array(desired_classes) if desired_classes is not None else None # if None the model gives you all classes
        self.t_inf = 0 
        self.preds = None
        self.unwanted_classes = None
        if desired_classes is not None and len(desired_classes)*2 > num_classes:
            self.unwanted_classes = np.array([x for x in range(num_classes) if x not in desired_classes])
        return None
    
    
    def _init_model(self, model_onnx_path: bytes) -> None:
        core = Core()
        model = core.read_model(model_onnx_path)
        model_compiled = core.compile_model(model, device_name="CPU", config={"PERFORMANCE_HINT":"LATENCY"})
        # .outputs is a list of outputs

        # Get the input and output nodes.
        keys_input = model_compiled.input(0) # has a field .any_name
        keys_output = model_compiled.output(0)
        
        # dtype = DTYPE_OPENVINO2NUMPY[keys_input_onnx.element_type.get_type_name()]
        # img_size = list(keys_input_onnx.shape)[2:]
        
        self.model_compiled = model_compiled
        self.keys_input = keys_input
        self.keys_output = keys_output        
        return None
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Assuming batch_size=1
        """
        t0 = perf_counter()
        res = self._infer(img)
        self.t_inf = perf_counter() - t0
        return res
    
    def get_duration_inference(self) -> float:
        return self.t_inf
    
    def _infer(self, img: np.ndarray) ->  np.ndarray:
        preds = self.model_compiled([img])[self.keys_output]
        self._set_preds(preds)
        self._filter_by_class_()
        return self.preds
    
    def _set_preds(self, preds: np.ndarray):
        # preds - output from inference
        self.preds = preds # [batch_id, x1, y1, x2, y2, cls_id, conf]
        return None
    
    def get_bboxes_areas(self, preds: np.ndarray) -> np.ndarray:
        # preds ~ [batch_id, x1, y1, x2, y2, cls_id, conf]
        bboxes = preds[:, 1:5]
        return (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    
    def _filter_by_class_(self) -> None:
        if self.desired_classes is None:
            return None
        if self.unwanted_classes is not None:
            self.preds = self.preds[~((self.preds[:, 5:6] == self.unwanted_classes).any(axis=1))]
            return None
        self.preds = self.preds[(self.preds[:, 5:6] == np.array(self.desired_classes)).any(axis=1)]
        return None