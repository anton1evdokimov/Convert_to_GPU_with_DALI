from typing import Optional, Dict, Tuple
from dataclasses import dataclass

import numpy as np


class ClassUnification:
    
    def __init__(self, rule_old2new: Dict[int, int], iou_thres: float = 0.45) -> None:
        self.rule_old2new = rule_old2new
        self.iou_thres = iou_thres
        return None
    
    
    def unite(self, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> Tuple[np.ndarray, np.ndarray , np.ndarray]:
        """ 
        Keyword arguments:
        boxes -- a NumPy array containing N bounding-box coordinates,
            with shape (N,4); 4 for x1,y1,x2,y2 coordinates of the boxes (xy1=top-left, xy2=bottom-right)
        classes -- a a NumPy array containing N class indexes corresponding to the bounding boxes, with shape N
        scores -- (Optional) a Numpy array containing the corresponding confidences, with shape N
        """
        final_boxes, final_scores, final_classes = [], [], []
        
        unaffected_classes = set(classes.tolist()) - (set(self.rule_old2new.keys()) | set(self.rule_old2new.values()))
        unaffected_classes = np.array([ind in unaffected_classes for ind in classes])
        final_boxes.append(boxes[unaffected_classes])
        final_scores.append(scores[unaffected_classes])
        final_classes.append(classes[unaffected_classes])
        
        classes_new = np.array([self.rule_old2new.get(ind.item(), ind.item()) for ind in classes])
        for ind in self.rule_old2new.values():
            mask = (classes_new == ind)
            selected_boxes = boxes[mask] 
            if selected_boxes.size == 0:
                continue
            selected_scores = scores[mask] 
            keep = self.nms(selected_boxes, selected_scores, self.iou_thres)
            
            final_boxes.append(selected_boxes[keep])
            final_scores.append(selected_scores[keep])
            final_classes.append(classes[mask][keep])
            
        return np.concatenate(final_boxes), np.concatenate(final_scores), np.concatenate(final_classes)
    
    
    @staticmethod
    def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
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
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            union = areas[i] + areas[order[1:]] - intersection
            iou = np.divide(intersection, union, where=union!=0)
            # we keep only those elements whose overlap
            # with the current bounding box is lower than the threshold:
            indexes = np.where(iou <= iou_thres)[0]
            # +1 because we had a shift (their indexes start from 0 but they should from 1)
            # when calculating xx1, yy1, ...
            order = order[indexes + 1] 
        return np.array(keep)