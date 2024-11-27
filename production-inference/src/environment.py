import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import platform

# from dotenv import load_dotenv

# from log.logger import get_logger

# logger = get_logger(__name__)


class LabelsModel(Enum):
    BACKPACK_OFF   = 0
    BACKPACK_ON    = 1
    BAG_OFF        = 2
    BAG_ON         = 3
    BOX            = 4
    COURIERBAG_OFF = 5
    SUITCASE       = 6

class LabelsCOCO(Enum):
    PERSON = 0
    BICYCLE = 1
    CAR = 2
    MOTORBIKE = 3
    AEROPLANE = 4
    BUS = 5
    TRAIN = 6
    TRUCK = 7
    BOAT = 8
    TRAFFIC_LIGHT = 9
    FIRE_HYDRANT = 10
    STOP_SIGN = 11
    PARKING_METER = 12
    BENCH = 13
    BIRD = 14
    CAT = 15
    DOG = 16
    HORSE = 17
    SHEEP = 18
    COW = 19
    ELEPHANT = 20
    BEAR = 21
    ZEBRA = 22
    GIRAFFE = 23
    BACKPACK = 24
    UMBRELLA = 25
    HANDBAG = 26
    TIE = 27
    SUITCASE = 28
    FRISBEE = 29
    SKIS = 30
    SNOWBOARD = 31
    SPORTS_BALL = 32
    KITE = 33
    BASEBALL_BAT = 34
    BASEBALL_GLOVE = 35
    SKATEBOARD = 36
    SURFBOARD = 37
    TENNIS_RACKET = 38
    BOTTLE = 39
    WINE_GLASS = 40
    CUP = 41
    FORK = 42
    KNIFE = 43
    SPOON = 44
    BOWL = 45
    BANANA = 46
    APPLE = 47
    SANDWICH = 48
    ORANGE = 49
    BROCCOLI = 50
    CARROT = 51
    HOT_DOG = 52
    PIZZA = 53
    DONUT = 54
    CAKE = 55
    CHAIR = 56
    SOFA = 57
    POTTEDPLANT = 58
    BED = 59
    DININGTABLE = 60
    TOILET = 61
    TVMONITOR = 62
    LAPTOP = 63
    MOUSE = 64
    REMOTE = 65
    KEYBOARD = 66
    CELL_PHONE = 67
    MICROWAVE = 68
    OVEN = 69
    TOASTER = 70
    SINK = 71
    REFRIGERATOR = 72
    BOOK = 73
    CLOCK = 74
    VASE = 75
    SCISSORS = 76
    TEDDY_BEAR = 77
    HAIR_DRIER = 78
    TOOTHBRUSH = 79

@dataclass
class CFGModel:
    IMG_SIZE: int = 640 # for yolov7 640 # 1280
    WEIGHTS: str = None
    NUM_CLASSES: int = 10
    DESIRED_LABELS: Optional[Tuple[str, ...]] = None # None for all classes 
    LABEL_THRES: Dict[str, float] = field(default_factory= lambda:{
        "PERSON"  : 0.5, # 0.4
        "BICYCLE"  : 0.5, # 0.4
        "CAR"  : 0.5, # 0.4
        "MOTORBIKE"  : 0.5, # 0.4
        "AEROPLANE"  : 0.5, # 0.4
        "BUS"  : 0.5, # 0.4
        "TRAIN"  : 0.5, # 0.4
        "TRUCK"  : 0.5, # 0.4
        "BOAT"  : 0.5, # 0.4
        "TRAFFIC_LIGHT"  : 0.5, # 0.4
    })
    DESIRED_CLASSES: Optional[List[int]] = None
    CLS2LABEL: LabelsCOCO = LabelsCOCO
    CLS_THRES: Dict[int, float] = None
    IS_END2END: bool = True # for tensorrt
    
    def __post_init__(self):
        self.CLS_THRES = {getattr(LabelsCOCO, label).value: conf for label, conf in self.LABEL_THRES.items()}
        if self.DESIRED_LABELS is not None:
            self.DESIRED_CLASSES: List[int] = [getattr(LabelsCOCO, label).value for label in self.DESIRED_LABELS]
        for p in Path('weights').iterdir():
            self.WEIGHTS = str(p)
        return None


@dataclass
class CFGTracking:
    pass
    # MAX_AGE: int = 45
    # MIN_HITS: int = 2
    # IOU_THRESHOLD: float = 0.3
    # ORDINAL_IND2CLASS_IND: Optional[Dict[int, int]] = None
    # CLS2LABEL: LabelsCOCO = LabelsCOCO
    # CLS_THRES: Dict[int, float] = None
    # LABEL_THRES: Dict[str, float] = field(default_factory= lambda:{
    #     'PERSON'   : 0.4,
    #     'CAR'      : 0.4,
    #     "TRUCK"    : 0.5,
    #     "BUS"      : 0.5,
    #     'MOTORBIKE': 0.4,
    #     'BACKPACK' : 0.2,
    #     "HANDBAG"  : 0.2,
    #     "SUITCASE" : 0.2
    # })
    
    # def __post_init__(self):
    #     self.CLS_THRES = {getattr(LabelsCOCO, label).value: conf for label, conf in self.LABEL_THRES.items()}
    #     return None

@dataclass
class CFGKafka:
    # consumer
    BOOTSTRAP_SERVERS : List[str] = None
    INPUT_TOPIC : str = None
    GROUP_ID    : str = None            #KAFKA_CONSUMER_GROUP
    ENABLE_AUTO_COMMIT: bool = False
    START_OFFSET: Optional[int] = None
    END_OFFSET  : Optional[int] = None
    CONSUMER_TIMEOUT_MS: Union[int, float] = 5_000 # float('inf')
    # CHECK_CRCS: bool = True
    # SESSION_TIMEOUT_MS: int
    # REQUEST_TIMEOUT_MS
    # MAX_PARTITION_FETCH_BYTES
    
    # producer
    OUTPUT_TOPIC: str = None
    BATCH_SIZE: int = 16384
    LINGER_MS: int = 0
    SHOULD_SEND_EMPTY = True
    HEADER_CONTENT_ENCODING: str = 'image_timestamp'
    
    def __post_init__(self):
        self.BOOTSTRAP_SERVERS = os.environ['KAFKA_BOOTSTRAP_SERVERS'].split(',')
        self.INPUT_TOPIC = os.environ['KAFKA_INPUT_TOPIC']
        self.OUTPUT_TOPIC = os.environ['KAFKA_OUTPUT_TOPIC']
        self.GROUP_ID = os.environ['KAFKA_CONSUMER_GROUP']
        if 'KAFKA_START_OFFSET' in os.environ:
            self.START_OFFSET = int(os.environ['KAFKA_START_OFFSET'])
        if 'KAFKA_END_OFFSET' in os.environ:
            self.END_OFFSET = int(os.environ["KAFKA_END_OFFSET"])
        if 'KAFKA_CONSUMER_TIMEOUT_MS' in os.environ:
            self.CONSUMER_TIMEOUT_MS = int(os.environ['KAFKA_CONSUMER_TIMEOUT_MS'])
        if 'KAFKA_BATCH_SIZE' in os.environ:
            self.BATCH_SIZE = int(os.environ['KAFKA_BATCH_SIZE'])
        if 'KAFKA_LINGER_MS' in os.environ:
            self.LINGER_MS = int(os.environ['KAFKA_LINGER_MS'])
        if 'KAFKA_SHOULD_SEND_EMPTY_RESULT' in os.environ:
            self.SHOULD_SEND_EMPTY = bool(os.environ['KAFKA_SHOULD_SEND_EMPTY_RESULT'])
        return None

@dataclass
class CFGMetrics:
    IS_ENABLED: bool = False
    ADDRESS: str = "localhost:4317"
    HOST_NAME: str = platform.node()
    
    def __post_init__(self):
        self.IS_ENABLED = bool(int(os.environ.get("ENABLE_METRICS", 0)))
        if "METRICS_ADDR" in os.environ:
            self.ADDRESS = os.environ['METRICS_ADDR']
        if "NEURAL_HOST_NAME" in os.environ:
            self.HOST_NAME = os.environ['NEURAL_HOST_NAME']
        return None


@dataclass
class CFGGlobal:
    MODEL    : CFGModel     = None
    TRACKING : CFGTracking  = None
    KAFKA    : CFGKafka     = None
    METRICS  : CFGMetrics   = None
    
    def __post_init__(self):
        self.MODEL     = CFGModel()
        self.TRACKING  = CFGTracking()
        # self.KAFKA     = CFGKafka()
        # self.METRICS   = CFGMetrics()
        # self.log()
        return None

    # def log(self):
    #     for cls_ in self.__dict__.values():
    #         logger.info(str(cls_))
    #     return None
