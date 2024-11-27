import uuid
from typing import Union, List, Optional, Dict, Tuple, Any
import json
import copy


class Point:
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        return None

    def update_coordinates(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        return None
    
    def dot(self, p) -> float:
        return self.x * p.x + self.y * p.y
    
    def int(self):
        self.x = int(self.x)
        self.y = int(self.y)
        return self
    
    def copy(self):
        return copy.deepcopy(self)
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    
    def __mul__(self, number):
        return Point(self.x * number, self.y * number)
    
    def __rmul__(self, other): #  other*self
        return self.__mul__(other)
    
    def __neg__(self): # -self
        return self * -1
    
    def __str__(self):
        return f'Point(x={self.x}, y={self.y})'
    
    def __repr__(self):
        return f'Point(x={self.x}, y={self.y})'


class Bbox:
    
    def __init__(self, x1: float, y1: float, x2: float, y2: float, label: Union[str, int], id_: Optional[int] = None, lastseen: Union[int, float] = 0, conf: float = -1):
        self.ul = Point(x1, y1) # upper left
        self.br = Point(x2, y2) # bottom right
        self.label = label
        self.lastseen = lastseen # used for detecting left items 
        self.birth_time = lastseen
        self.id_ = id_ # id from tracking
        self.conf = conf 
        self._intersect_state = '' # the state of line intersection. In order to count intersections, one should intersect two lines 
        self._area_state = '' # the state of area intersection. -//-
        # self.uuid = str(uuid.UUID(int=self.id_))
        self.home_position = (self.ul.copy(), self.br.copy())

        # self.is_passenger = False
        # self.is_intersected = False
        self.motionless = False
        self.nframe_motionless = 0
        return None
    
    def update_by(self, bbox: "Bbox"):
        self.update_coordinates(bbox)
        self.update_lastseen(bbox.lastseen)
        return None
    
    
    def update_coordinates(self, xyxy: Union[Tuple[int, int, int ,int], "Bbox"]) -> None:
        if isinstance(xyxy, Bbox):
            self.ul.update_coordinates(xyxy.ul.x, xyxy.ul.y)
            self.br.update_coordinates(xyxy.br.x, xyxy.br.y)
        else:
            self.ul.update_coordinates(xyxy[0], xyxy[1])
            self.br.update_coordinates(xyxy[2], xyxy[3])
        
        difs = (self.ul - self.home_position[0], self.br - self.home_position[1])
        scale_eff =  0.07 * (self.get_h()**2 + self.get_w()**2)**(1/2) # 7% of the diagonal of a bbox
        if all(p.dot(p)**(1/2) > scale_eff for p in difs):
            self.home_position = (self.ul.copy(), self.br.copy())
            self.nframe_motionless = 0
        else:
            self.nframe_motionless += 1
        return None
    
    def get_center(self) -> Point:
        return Point((self.ul.x+self.br.x) / 2, (self.ul.y+self.br.y) / 2)
    
    def get_bottom_center(self) -> Point:
        return Point((self.ul.x + self.br.x) / 2, self.br.y)
    
    def get_area(self) -> int:
        return (self.br.x - self.ul.x) * (self.br.y - self.ul.y)
    
    def get_w(self):
        return self.br.x - self.ul.x
    
    def get_h(self):
        return self.br.y - self.ul.y
    
    def update_lastseen(self, time: float) -> None:
        self.lastseen = time
        return None
    
    def set_conf(self, conf: float):
        # should be in the range of [0, 1]
        self.conf = conf
        return None
    
    def set_intersect_state(self, state: str):
        self._intersect_state = state
        return None
    
    def get_intersect_state(self) -> str:
        return self._intersect_state
    
    def set_area_state(self, state: str):
        self._area_state = state
        return None
    
    def get_area_state(self) -> str:
        return self._area_state
    
    def copy(self):
        return copy.deepcopy(self)


class Frame:
    
    def __init__(self, timestamp: Union[int, float], kafka_key: Optional[Any] = None, bboxes: Optional[Union[Bbox, List[Bbox]]] = None):
        self.timestamp = timestamp
        self.kafka_key = kafka_key
        self.bboxes: List[Bbox] = []
        if bboxes is not None:
            self.add_bboxes(bboxes)
        return None
    
    def add_bboxes(self, bboxes: Union[Bbox, List[Bbox]]) -> None:
        if not bboxes:
            return None
        if isinstance(bboxes, Bbox):
            bboxes = [bboxes]
        self.bboxes += bboxes
        return None
    
    def get_kafka_key(self):
        return self.kafka_key
    
    def is_empty(self):
        return len(self.bboxes) == 0
    
    def get_num_results(self):
        return len(self.bboxes)
    
    def form_json(self) -> str:
        bBoxes = []
        for bbox in self.bboxes:
            res = {
                'label'   : bbox.label,
                'x0'      : float(bbox.ul.x),
                'y0'      : float(bbox.ul.y),
                'x1'      : float(bbox.br.x),
                'y1'      : float(bbox.br.y),
            }
            if bbox.conf >= 0:
                res['confidence'] = float(bbox.conf)
            if bbox.id_ is not None:
                res['objectId'] = str(uuid.UUID(int=bbox.id_))
            bBoxes.append(res)
        result = {
            'bBoxes': bBoxes,
        }
        return json.dumps(result) 
