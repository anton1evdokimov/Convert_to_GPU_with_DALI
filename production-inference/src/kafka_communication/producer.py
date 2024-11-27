from dataclasses import dataclass
from typing import List, Tuple, Optional

from kafka import KafkaProducer

from environment import CFGKafka
from log.logger import get_logger


logger = get_logger(__name__)


@dataclass
class KafkaProducerInput:
    value: bytes
    key: Optional[str] = None
    headers: Optional[List[Tuple[str, bytes]]] = None


class KafkaMsgProducer:
    def __init__(self, cfg: CFGKafka) -> None:
        self.cfg = cfg
        self.producer = KafkaProducer(
            bootstrap_servers = cfg.BOOTSTRAP_SERVERS,
            batch_size = cfg.BATCH_SIZE,
            linger_ms = cfg.LINGER_MS,
        )
        return None
    
    def send(self, msg: KafkaProducerInput) -> None:
        try:
            self.producer.send(self.cfg.OUTPUT_TOPIC, value=msg.value, headers=msg.headers, key=msg.key)
        except Exception as e:
            logger.warning(f'Unable to send the message to {self.cfg.OUTPUT_TOPIC}: {e}', exc_info=True)
        return None
    
    def should_send_empty(self):
        return self.cfg.SHOULD_SEND_EMPTY
    
    def get_header_content_encoding(self):
        return self.cfg.HEADER_CONTENT_ENCODING
    
    def flush(self) -> None:
        self.producer.flush(timeout=5.)
        return None