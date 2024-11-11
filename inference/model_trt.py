
import time
import cupy as cp
from pathlib import Path
from typing import List, List, Optional, Tuple, Union


import tensorrt as trt
import numpy as np
import pycuda.autoinit  # requires to prevent <<pycuda._driver.LogicError: explicit_context_dependent failed: invalid device context - no currently active context?>>
import pycuda.driver as cuda


class ModelTRT:
    
    def __init__(self, serialized_engine: bytes, num_total_classes: int, desired_classes: Optional[List[int]], is_end2end: bool,
                iou_thres: Optional[float] = None, 
                score_thres: Optional[float] = None, 
                ):
        """
        This model is expected to be serialized via:
        
        1) ./yolov7/export.py --grid --end2end --simplify ...
        Example: python export.py --weights ./yolov7-tiny.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
        
        # Export TensorRT-engine model 
        python export.py -o ./yolov7-end2end.onnx -e ./yolov7-end2end.trt -p fp16

        2) ./yolov7/export.py --grid  --simplify
        Example: python export.py --weights ./yolov7-tiny.pt --grid --simplify
        
        """
        self.desired_classes = desired_classes
        self.num_total_classes = num_total_classes
        self.is_end2end = is_end2end
        self.t_inf = 0
        self.unwanted_classes = None
        if desired_classes is not None and len(desired_classes)*2 > num_total_classes:
            self.unwanted_classes = cp.array([x for x in range(num_total_classes) if x not in desired_classes])


        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins

        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.img_size = engine.get_binding_shape(0)[2:]  # in order to call assert to check img-size
        if self.is_end2end:
            self.num_max_dets = engine.get_binding_shape("det_scores")[-1]
            # self.num_max_dets = engine.get_binding_shape("scores")[-1]
        else:
            args_required = (iou_thres, score_thres)
            assert all(arg is not None for arg in args_required), f'All arguments {args_required} must be provided'
            self.iou_thres = iou_thres
            self.score_thres = score_thres
            self.batch_size = engine.get_binding_shape(0)[0]
        self.context = engine.create_execution_context()
        self.stream = cuda.Stream()
        
        self.inputs, self.outputs, self.bindings = [], [], []
        for binding in engine:
            # Determine dimensions and create page-locked memory buffers (which won't be swapped to disk) to hold host inputs/outputs.
            size = trt.volume(engine.get_tensor_shape(binding))
            dtype = trt.nptype(engine.get_tensor_dtype(binding))
            # Allocate a pagelocked numpy.ndarray (so that GPU could transfer data to the main memory without the involvement of the CPU)
            host_mem = cuda.pagelocked_empty(size, dtype)                                 
            # Allocate device memory for inputs and outputs (the same size as host' input and output).
            device_mem = cuda.mem_alloc(host_mem.nbytes) # Objects of this type (DeviceAllocation) can be cast to int to obtain a linear index into this Context’s memory
            self.bindings.append(int(device_mem))
            mem_allocations = {
                'host'  : host_mem,
                'device': device_mem
            }
            if engine.binding_is_input(binding):
                self.inputs.append(mem_allocations)
            else:
                self.outputs.append(mem_allocations)
        return None
    
    def get_duration_inference(self) -> float:
        return self.t_inf
    
    def __call__(self, img: cp.ndarray) -> Tuple[List[cp.ndarray], List[cp.ndarray], List[cp.ndarray]]:
        """
        Returns the Tuple of Lists (final_boxes, final_scores, final_cls_inds)
        final_boxes - final bboxes for objects, shape ~ (num_dets, 4) (xyxy);
        
        final_scores - the corresponding scores of bboxes, shape ~ (num_dets,);    
        
        final_cls_inds - the corresponding class indexes of bboxes, shape ~ (num_dets,) 
        """
        t0 = time.perf_counter()
        preds = self._infer(img)
        if self.is_end2end:
            res = self._postprocess(preds)
        else:
            res = self._postprocess_nms(preds)
        self.t_inf = time.perf_counter() - t0
        return res
    
    def _infer(self, img: cp.ndarray) -> List[cp.ndarray]:
        cp.copyto(self.inputs[0]['host'], img.ravel())
        # Transfer input data from CPU to GPU.
        for input_ in self.inputs:
            # Copy from the Python buffer src to the device pointer dest (an int or a DeviceAllocation)
            # src must be page-locked memory
            cuda.memcpy_htod_async(input_['device'], input_['host'], self.stream)
        # infer
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        # Transfer data GPU back to CPU
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
        self.stream.synchronize()
        return [output['host'] for output in self.outputs]
    
    def _postprocess(self, preds: List[cp.ndarray]) -> Tuple[List[cp.ndarray], List[cp.ndarray], List[cp.ndarray]]:
        """
        preds is a list because the model might have a number of different outputs (not only 1) 
        Now we assume that there is only 1 output of the model (tested for batch_size=1)
        
        Network Description (Example when exporting with --include-nms or --end2end in yolov7/export.py)
        Input 'images' with shape (1, 3, 640, 640) and dtype DataType.FLOAT
        Output 'num_dets' with shape (1, 1) and dtype DataType.INT32
        Output 'det_boxes' with shape (1, 100, 4) and dtype DataType.FLOAT (xyxy)
        Output 'det_scores' with shape (1, 100) and dtype DataType.FLOAT
        Output 'det_classes' with shape (1, 100) and dtype DataType.INT32
        """
        num_dets, boxes, scores, cls_inds = preds # batch is the first dim (output ~ (1,) (400,) (100,) (100,))
        num_dets, boxes, scores, cls_inds = num_dets.reshape(-1, 1), boxes.reshape(-1, self.num_max_dets, 4), scores.reshape(-1, self.num_max_dets), cls_inds.reshape(-1, self.num_max_dets)
        final_boxes, final_scores, final_cls_inds = [], [], []
        for i, num in enumerate(num_dets):
            num = num.item()
            cls_ind = cls_inds[i, :num]
            mask = self._filter_by_classes(cls_ind)
            final_boxes.append(boxes[i, :num, :][mask])
            final_scores.append(scores[i, :num][mask])
            final_cls_inds.append(cls_ind[mask])
        return final_boxes, final_scores, final_cls_inds
    
    def _postprocess_nms(self, preds: List[cp.ndarray]) -> Tuple[List[cp.ndarray], List[cp.ndarray], List[cp.ndarray]]:
        """
        -//-
        (Without --include-nms or --end2end)
        Output 'output' with shape (batch_size, num_dets, num_classes) and dtype DataType.FLOAT
        """
        
        preds = preds[0].reshape(self.batch_size, -1, 5 + self.num_total_classes)
        batch_boxes = self.xywh2xyxy(preds[..., :4])
        batch_scores = preds[..., 4:5] * preds[..., 5:]
        if self.desired_classes is None:
            self.desired_classes = list(range(self.num_total_classes))
        
        final_boxes, final_scores, final_cls_inds = [], [], []
        for boxes, scores in zip(batch_boxes, batch_scores):
            boxes, scores, cls_inds = self.multiclass_nms(boxes, scores, self.desired_classes, self.iou_thres, self.score_thres)
            final_boxes.append(boxes)
            final_scores.append(scores)
            final_cls_inds.append(cls_inds)
        return final_boxes, final_scores, final_cls_inds

    def _filter_by_classes(self, cls_inds: cp.ndarray) -> cp.ndarray:
        if self.desired_classes is None:
            return cp.ones(cls_inds.size, dtype=bool)
        if self.unwanted_classes is not None:
            mask = ~((cls_inds.reshape(-1, 1)  == self.unwanted_classes).any(axis=1))
            return mask
        mask = (cls_inds.reshape(-1, 1) == cp.array(self.desired_classes)).any(axis=1)
        return mask

    
    def assert_correct_img_size(self, img_size) -> None:
        error_msg = f'The target img size {img_size} does not match the serialised size of the model {self.img_size}'
        assert img_size[0] == self.img_size[0] and img_size[1] == self.img_size[1], error_msg
        return None
    
    @staticmethod
    def nms(boxes: cp.ndarray, scores: cp.ndarray, iou_thres: float) -> cp.ndarray:
        """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding boxes with their
        confidence scores and return an array with the indexes of the bounding boxes we want to
        keep (and display later).

        Keyword arguments:
        boxes -- a NumPy array containing N bounding-box coordinates,
        with shape (N,4); 4 for x1,y1,x2,y2 coordinates of the boxes (xy1=top-left, xy2=bottom-right)
        scores -- a Numpy array containing the corresponding confidences with shape N
        """
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            # Index of the current element:
            i = order[0]
            keep.append(i)
            
            # calculate iou between current bbox and all other candidates
            xx1 = cp.maximum(x1[i], x1[order[1:]])
            yy1 = cp.maximum(y1[i], y1[order[1:]])
            xx2 = cp.minimum(x2[i], x2[order[1:]])
            yy2 = cp.minimum(y2[i], y2[order[1:]])
            
            w = cp.maximum(0, xx2 - xx1)
            h = cp.maximum(0, yy2 - yy1)
            intersection = w * h
            union = areas[i] + areas[order[1:]] - intersection
            iou = cp.divide(intersection, union, where=union!=0)
            # we keep only those elements whose overlap
            # with the current bounding box is lower than the threshold:
            indexes = cp.where(iou <= iou_thres)[0]
            # +1 because we had a shift (their indexes start from 0 but they should from 1)
            # when calculating xx1, yy1, ...
            order = order[indexes + 1] 
        return cp.array(keep)
    
    @staticmethod
    def multiclass_nms(boxes: cp.ndarray, scores: cp.ndarray, desired_classes: List[int], iou_thres: float, score_thres: float) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        Keyword arguments:
        boxes -- a NumPy array containing N bounding-box coordinates,
        with shape (N,4); 4 for x1,y1,x2,y2 coordinates of the boxes (xy1=top-left, xy2=bottom-right)
        scores -- a Numpy array containing the corresponding confidences
        
        boxes.shape ~ (num_dets, 4);
        scores.shape ~ (num_dets, num_classes)
        """
        final_boxes, final_scores, final_classes = [cp.empty((0, 4), dtype=cp.float32)], [cp.empty(0, dtype=cp.float32)], [cp.empty(0, dtype=cp.int32)]
        for cls_ in desired_classes:
            cls_scores = scores[:, cls_]
            mask_thres = cls_scores >= score_thres
            
            scores_valid = cls_scores[mask_thres]
            if scores_valid.size == 0:
                continue
            boxes_valid = boxes[mask_thres]
            keep = ModelTRT.nms(boxes_valid, scores_valid, iou_thres)
            final_boxes.append(boxes_valid[keep])
            final_scores.append(scores_valid[keep])
            final_classes.append(cp.ones(len(keep), dtype=cp.int32) * cls_)
        return cp.concatenate(final_boxes), cp.concatenate(final_scores), cp.concatenate(final_classes)
    
    @staticmethod
    def xywh2xyxy(boxes_wh: cp.ndarray) -> cp.ndarray:
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        boxes_xyxy = cp.empty(boxes_wh.shape, dtype=boxes_wh.dtype)
        boxes_xyxy[..., 0] = boxes_wh[..., 0] - boxes_wh[..., 2] / 2 # top left x 
        boxes_xyxy[..., 1] = boxes_wh[..., 1] - boxes_wh[..., 3] / 2 # top left y
        boxes_xyxy[..., 2] = boxes_wh[..., 0] + boxes_wh[..., 2] / 2 # bottom right x
        boxes_xyxy[..., 3] = boxes_wh[..., 1] + boxes_wh[..., 3] / 2 # bottom right y
        return boxes_xyxy
    
    # @staticmethod
    # def xywh2xyxy(boxes_wh: cp.ndarray) -> cp.ndarray:
    #     """
    #     this approach turns out to be slower for the numpy implementation 
    #     """
    #     xywh2xyxy_matrix = cp.array([
    #         [1, 0 , 1, 0],
    #         [0, 1 , 0, 1],
    #         [-0.5, 0, 0.5, 0],
    #         [0, -0.5, 0, 0.5]],
    #         dtype=boxes_wh.dtype)
    #     return boxes_wh @ xywh2xyxy_matrix
