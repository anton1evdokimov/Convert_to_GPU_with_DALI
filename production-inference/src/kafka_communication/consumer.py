import multiprocessing
import time
from dataclasses import dataclass
# from multiprocessing.connection import PipeConnection
from typing import Callable, NoReturn, Optional, Tuple, Any, List

import numpy as np
import cv2
from kafka import KafkaConsumer
from kafka.consumer.fetcher import ConsumerRecord
from kafka.structs import TopicPartition

from environment import CFGKafka
from log.logger import get_logger

logger = get_logger(__name__)


@dataclass
class KafkaConsumerOutput:
    timestamp: int
    img: Optional[np.ndarray]
    key: str # not sure
    error: Optional[Tuple[str, ConsumerRecord]]
    


class KafkaMsgConsumer(multiprocessing.Process):
    
    def __init__(self, cfg: CFGKafka, process_msg: Optional[Callable[[ConsumerRecord], KafkaConsumerOutput]] = None):
        super().__init__()
        self.cfg = cfg
        self.process_msg = process_msg if process_msg is not None else self.process_msg
        
        self.stop_event = multiprocessing.Event()
        self.queue = multiprocessing.Queue(maxsize=100) # maxsize number of items
        self.started = multiprocessing.Event()
        return None
    
    def stop(self) -> None:
        """
        The kafka process will not be fully terminated util the self.queue is not empty.
        If to call join() from the main process, it will hang indefinitely in case self.queue is not empty
        After draining the self.queue the kafka process will stop
        """
        logger.info('Shutting down a kafka consumer')
        self.stop_event.set()
        return None
    
    def get_msg_queue(self):
        return self.queue
    
    def is_started(self):
        return self.started.is_set()
    
    def run(self) -> NoReturn:
        logger.info('Connecting to Kafka')
        try:
            consumer = KafkaConsumer(
                bootstrap_servers=self.cfg.BOOTSTRAP_SERVERS,
                group_id=self.cfg.GROUP_ID,
                auto_offset_reset='earliest',
                enable_auto_commit=False,
                consumer_timeout_ms=self.cfg.CONSUMER_TIMEOUT_MS,
                metadata_max_age_ms=20)
            consumer.subscribe(pattern=self.cfg.INPUT_TOPIC)
            
            self.started.set()
            logger.info('Connected to Kafka')
            while not self.stop_event.is_set():
                for msg in consumer:
                    # if self.cfg.START_OFFSET is not None:
                    #     print(self.consumer.assignment())
                    #     self.consumer.seek([TopicPartition(topic=self.cfg.INIT_TOPIC[0], partition=0)], self.cfg.START_OFFSET)
                    #     break
                    if self.stop_event.is_set():
                        break
                    try: 
                        consumer.commit()
                    except Exception as e:
                        logger.warning(f'Could not commit a message: {str(e)}', exc_info=True)
                        time.sleep(1.)
                        continue
                    logger.debug(f"Received message {msg.timestamp}")
                    res = self.process_msg(msg)
                    self.queue.put(res)
                #! think about offsets to stop processing
                else:
                    if self.cfg.END_OFFSET is not None:
                        break
            consumer.close()
        except Exception as e:
            logger.error(e, exc_info=True)
            consumer.close()
        finally:
            self.queue.put(None)
            self.queue.close()
            logger.info('Stopped receiving massages from Kafka')
            
    
    @staticmethod
    def process_msg(msg: ConsumerRecord) -> KafkaConsumerOutput:
        error = None
        img_bytes = msg.value
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            error = ('Could not decode an image', msg)
        return KafkaConsumerOutput(timestamp=msg.timestamp, img=img, key=msg.key, error=error)