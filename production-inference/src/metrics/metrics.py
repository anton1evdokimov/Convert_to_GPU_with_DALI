import os
from typing import Any, Optional
import platform

from opentelemetry import metrics
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

from log.logger import get_logger
from environment import CFGMetrics

logger = get_logger(__name__)


class MetricsRecorder:
    
    def __init__(self, cfg: CFGMetrics):
        if not cfg.IS_ENABLED:
            return None
        self.cfg = cfg
        
        try:
            resource = Resource(attributes={
                                    SERVICE_NAME: "geometry-nn",
                                    "host_name": self.cfg.HOST_NAME
                                })
            metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=self.cfg.ADDRESS, insecure=True), export_interval_millis=15000)
            provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
            metrics.set_meter_provider(provider)
            meter = metrics.get_meter("Neural_network")
            # https://opentelemetry.io/docs/specs/otel/metrics/api/#histogram-creation
            self.inference_duration     = meter.create_histogram(name='inference.duration', description='Inference duration (inference + NMS + tracking)', unit='ms')
            self.preprocessing_duration = meter.create_histogram(name='preprocessing.duration', description='Preprocessing duration', unit='ms')
            self.working_cycle_duration = meter.create_histogram(name='working_cycle.duration', description='Duration of the whole cycle (decoding + Preprocessing + Inference + Sending)', unit='ms')
        except Exception as e:
            logger.error(str(e), exc_info=True)
        return None
    
    def record_inference(self, seconds: float):
        if not self.cfg.IS_ENABLED:
            return None
        self._record_histogram(self.inference_duration, int(seconds * 1000))
        return None
    
    def record_preprocessing(self, seconds: float):
        if not self.cfg.IS_ENABLED:
            return None
        self._record_histogram(self.preprocessing_duration, int(seconds * 1000))
        return None
    
    def record_one_working_cycle(self, seconds: float):
        if not self.cfg.IS_ENABLED:
            return None
        self._record_histogram(self.working_cycle_duration, int(seconds * 1000))
        return None
    
    def _record_histogram(self, histogram, value: Any):
        try:
            histogram.record(value, {"host_name": self.cfg.HOST_NAME}) # adding flexible attributes to dict
        except Exception as e:
            logger.error(str(e), exc_info=True)
        return None

