import signal
from typing import List
from time import time, sleep
from queue import Empty
import datetime

from .consumer import KafkaMsgConsumer, KafkaConsumerOutput
from .producer import KafkaMsgProducer, KafkaProducerInput
from analytics.bbox import Frame
from log.logger import get_logger


logger = get_logger(__name__)


class GracefulKiller():
    kill_now = False
    def __init__(self) -> None:
        signal.signal(signal.SIGINT, self._exit_gracefully)
        signal.signal(signal.SIGTERM, self._exit_gracefully)

    def _exit_gracefully(self, *args, **kwargs)-> None:
        self.kill_now = True


class Orchestrator:
    
    def __init__(self, consumer: KafkaMsgConsumer, producer: KafkaMsgProducer):
        self.consumer = consumer
        self.producer = producer
        self.killer = GracefulKiller()
        
        self.queue = self.consumer.get_msg_queue()
        self.consumer.start()
        self.stopped = False
        self._wait_until_consumer_starts()
        return None
    
    def _wait_until_consumer_starts(self, timeout: float=10.):
        end_time = time() + timeout
        while time() < end_time:
            if self.consumer.is_started():
                return None
            sleep(0.1)
        raise ConnectionError(f'Could not connect to Kafka {self.consumer.cfg}')
    
    def __iter__(self):
        return self
    
    def __next__(self) -> KafkaConsumerOutput:
        while True:
            if self.killer.kill_now and not self.stopped:
                self.consumer.stop()
                self.stopped = True
            try:
                msg: KafkaConsumerOutput = self.queue.get(timeout=5.)
            except Empty:
                logger.info('Pending images')
                continue
            if msg is None:
                self.producer.flush()
                logger.info('All messages from the inner buffer were processed')
                raise StopIteration
            elif msg.error is not None:
                logger.warning(msg.error)
                continue
            return msg
    
    def send_result(self, result: Frame) -> None:
        if not self.producer.should_send_empty() and result.is_empty():
            return None
        value  = result.form_json().encode()
        headers = [(self.producer.get_header_content_encoding(), self.time_unix2rfc3339(result.timestamp/1000).encode())]
        msg = KafkaProducerInput(value, key=result.get_kafka_key(), headers=headers)
        
        logger.info(f'Timestamp {result.timestamp}: Sending {result.get_num_results()} bboxes')
        self.producer.send(msg)
        return None
    
    @staticmethod
    def time_unix2rfc3339(timestamp: float):
        return datetime.datetime.fromtimestamp(timestamp).astimezone().isoformat()