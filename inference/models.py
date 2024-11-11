import cupy as cp
from time import perf_counter
from typing import List, Dict, Tuple

from dataloader.preprocessing import ImgPipeline
from analytics.bbox import Bbox
from environment import CFGTracking, CFGModel
from log.logger import get_logger
from .sort import Sort as Tracker

logger = get_logger(__name__)


class ModelDetect:
    
    def __init__(self, model, pipeline: ImgPipeline) -> None:
        self.model = model
        self.pipeline = pipeline
        self.t_prep = 0
        self.t_inf = 0
        return None
    
    def __call__(self, img: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        t0 = perf_counter()
        img = self.pipeline.process(img)
        t1 = perf_counter()
        self.t_prep = t1 - t0
        
        t0 = perf_counter()
        output = self.model(img) 
        if isinstance(output, cp.ndarray):
            output = (output[:, 1:5], output[:, 6], output[:, 5]) # (bboxes, scores, cls_inds)
        else: # for tensorrt models (List[cp.ndarray], List[cp.ndarray], List[cp.ndarray])
            output = tuple(x[0] for x in output)
        self.pipeline.coords_unscale(output[0])
        t1 = perf_counter()
        self.t_inf = t1 - t0
        return output # (bboxes, scores, cls_inds)
    
    def get_duration_preprocessing(self):
        return self.t_prep
    
    def get_duration_inference(self):
        return self.t_inf
    
    @staticmethod
    def preds2bboxes(boxes: cp.ndarray, scores: cp.ndarray, cls_inds: cp.ndarray, cfg: CFGModel) -> List[Bbox]:
        results = []
        for xyxy, conf, cls_ in zip(boxes, scores, cls_inds):
            if cls_ not in cfg.CLS_THRES or conf < cfg.CLS_THRES[cls_]:
                continue
            results.append(Bbox(*xyxy, label=cfg.CLS2LABEL(int(cls_)).name, conf=conf.item()))
        return results


class ModelTracking:
    
    def __init__(self, model_detect: ModelDetect, cfg: CFGTracking):
        super().__init__()
        self.cfg = cfg
        self.model_detect = model_detect
        self.trackers = {cls_ind: Tracker(max_age=self.cfg.MAX_AGE, min_hits=self.cfg.MIN_HITS, iou_threshold=self.cfg.IOU_THRESHOLD) for cls_ind in self.cfg.CLS_THRES.keys()}
        
        self.t_prep = 0
        self.t_inf = 0
        return None
    
    def _preds_track(self, bboxes: cp.ndarray, scores: cp.ndarray, cls_inds: cp.ndarray) -> Dict[int, cp.ndarray]:
        objects_trackable = {cls_ind: cp.empty((0,5), dtype=cp.float32) for cls_ind in self.trackers}
        if len(bboxes):  # returns the size of the first dimension
            for xyxy, conf, cls_ in zip(bboxes, scores, cls_inds):
                if cls_ not in self.trackers or conf < self.cfg.CLS_THRES[cls_]:
                    continue
                if isinstance(objects_trackable[cls_], list):
                    # [x1, y1, x2, y2, conf]
                    # x1, y1 - coordinates of upper left, x2, y2 - coordinates of bottom right
                    objects_trackable[cls_].append([*xyxy, conf.item()])
                else:
                    objects_trackable[cls_] = [[*xyxy, conf.item()]]
            for cls_ind, bbox in objects_trackable.items():
                if isinstance(bbox, list):
                    objects_trackable[cls_ind] = cp.array(bbox, dtype=cp.float32)
    
        objects_tracked = {}
        for cls_ind, tracker in self.trackers.items():
            objects_tracked[cls_ind] = tracker.update(objects_trackable[cls_ind])
            # each line in tracks has x_left_upper, y_left_upper, x_right_bottom, y_right_bottom, id_of_object_from_SORT
        return objects_tracked
    
    def _tracked_tensor2bboxes(self, tracked_bboxes: Dict[int, cp.ndarray]) -> List[Bbox]:
        bboxes = []
        if not tracked_bboxes:
            return bboxes
        for cls_, bbox in tracked_bboxes.items():
            for *xyxy, obj_id in bbox:
                if self.cfg.ORDINAL_IND2CLASS_IND is not None:
                    cls_ = self.cfg.ORDINAL_IND2CLASS_IND[int(cls_)]
                bboxes.append(Bbox(*xyxy, id_=int(obj_id), label=self.cfg.CLS2LABEL(cls_).name))
        return bboxes
    
    def get_duration_preprocessing(self):
        return self.model_detect.get_duration_preprocessing()
    
    def get_duration_inference(self):
        return self.t_inf
    
    def __call__(self, img: cp.ndarray) -> List[Bbox]:
        t0 = perf_counter()
        preds = self.model_detect(img)
        objects_tracked = self._preds_track(*preds)
        # after tracking some coordinates might become negative
        for tracked_bboxes in objects_tracked.values():
            self.model_detect.pipeline.coords_clip(tracked_bboxes[:, :4])
        t1 = perf_counter()
        self.t_inf = t1 - t0
        return self._tracked_tensor2bboxes(objects_tracked)
    