import pycuda.driver as cuda # NOTE: One must import PyCuda driver first, before CVCUDA or VPF otherwise
                            # things may throw unexpected errors.
# import pycuda.autoinit  # requires to prevent <<pycuda._driver.LogicError: explicit_context_dependent failed: invalid device context - no currently active context?>>
import tensorrt as trt
import numpy as np
import time

import os
os.system("cd ../weights && ls")

class TensorRTInference:
    def __init__(self, model_params):
        """
            Return predictions in format:
            [left, top, width, height, confidence, class_id]
        """
        self.model_params = model_params
        self.trt_version = self._get_trt_version()
        
        self.engine_path = self.model_params["weights"]
        self.classes_conf = self.model_params["classes_ids_conf"]
        self.iou_thres = self.model_params["iou_thres"]
        self.infer_type = self.model_params["infer_type"]
        print(f'Loading TensorRT engine: {self.engine_path}, TensorRT version: {self.trt_version}')
        # logger.debug(f'Loading TensorRT engine: {self.engine_path}, TensorRT version: {self.trt_version}')

        # Load TRT engine
        self.TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(None, "")
        # trt.init_libnvinfer_plugins(self.TRT_LOGGER, namespace="")

        if self.trt_version == 8:
            self.engine, self.context = self._load_engine_v8(self.engine_path)
            print("self.context", self.context)
        elif self.trt_version == 10:
            self.engine, self.context = self._load_engine_v10(self.engine_path)
        else:
            raise ValueError(f"Unsupported TensorRT version: {self.trt_version}")

        assert self.engine, "Failed to load the engine."
        assert self.context, "Failed to create execution context."

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        self.batch_size = None
        self._setup_bindings()

    def _get_trt_version(self):
        version = int(trt.__version__.split('.')[0])
        return version

    def _clear_predictions(self):
        self.predictions = []

    def _clear_outputs(self):
        self.outputs = []

    def _load_engine_v8(self, engine_path):
        with open(engine_path, 'rb') as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        return engine, context

    def _load_engine_v10(self, engine_path):
        with open(engine_path, 'rb') as f:
            serialized_engine = f.read()
        runtime = trt.Runtime(self.TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        context = engine.create_execution_context()
        return engine, context

    def _setup_bindings(self):
        if self.trt_version == 8:
            for i in range(self.engine.num_bindings):
                is_input = self.engine.binding_is_input(i)
                name = self.engine.get_binding_name(i)
                dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
                shape = self.context.get_binding_shape(i)

                if is_input and shape[0] < 0:
                    assert self.engine.num_optimization_profiles > 0, "No optimization profiles available."
                    profile_shape = self.engine.get_profile_shape(0, name)
                    self.context.set_binding_shape(i, profile_shape[2]) # set max shape
                    shape = self.context.get_binding_shape(i)

                if is_input:
                    self.batch_size = shape[0]

                allocation_size = dtype.itemsize
                for s in shape:
                    allocation_size *= s
                allocation = cuda.mem_alloc(allocation_size)
                host_allocation = None if is_input else np.zeros(shape, dtype)

                binding = {
                    'index': i,
                    'name': name,
                    'dtype': dtype,
                    'shape': list(shape),
                    'allocation': allocation,
                    'host_allocation': host_allocation,
                }
                self.allocations.append(allocation)

                if is_input:
                    self.inputs.append(binding)
                else:
                    self.outputs.append(binding)

        elif self.trt_version == 10:
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                dtype = self.engine.get_tensor_dtype(name)
                shape = self.engine.get_tensor_shape(name)
                is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

                if is_input:
                    self.batch_size = shape[0]

                size = np.dtype(trt.nptype(dtype)).itemsize
                for s in shape:
                    size *= s

                allocation = cuda.mem_alloc(size)
                host_allocation = None if is_input else np.zeros(shape, np.dtype(trt.nptype(dtype)))
                binding = {
                    'index': i,
                    'name': name,
                    'dtype': np.dtype(trt.nptype(dtype)),
                    'shape': list(shape),
                    'allocation': allocation,
                    'host_allocation': host_allocation,
                    'size': size
                }
                self.allocations.append(allocation)
                if is_input:
                    self.inputs.append(binding)
                else:
                    self.outputs.append(binding)

        assert self.batch_size > 0, "Batch size must be greater than 0."
        assert self.inputs, "No input bindings found."
        assert self.outputs, "No output bindings found."

    def _filter_detect_predictions(self, preds):
        if self.trt_version == 10:
            num_dets, det_boxes, det_scores, det_classes = preds[:4]
            for pred_idx in range(num_dets.tolist()[0][0]):
                idx_class = int(det_classes[0, pred_idx])
                idx_score = det_scores[0, pred_idx]
                idx_box = [max(0, value) for value in det_boxes[0, pred_idx].tolist()]

                if idx_score > self.classes_conf[idx_class]:
                    self.predictions.append(idx_box + [idx_score, idx_class])

        elif self.trt_version == 8:
            preds=preds[0]
            # preds = non_max_suppression_cpu(prediction=preds[0],
            #                                 conf_thres=min(self.classes_conf.values()),
            #                                 iou_thres=self.iou_thres)
            for idx, box in enumerate(preds):
                idx_class = int(box[-1])
                idx_score = box[-2]
                idx_box = [max(0, value) for value in box[:4].tolist()]
                if idx_score > self.classes_conf[idx_class]:
                    self.predictions.append(idx_box + [idx_score, idx_class])
                # print("self.predictions", self.predictions)

    # def _filter_segm_predictions(self, preds):
    #     object_output = preds[1][0]

    #     all_boxes = object_output[:4, :].T

    #     confidences = object_output[4:12, :].T

    #     max_confidences = np.max(confidences, axis=1)
    #     max_conf_classes = np.argmax(confidences, axis=1)

    #     all_boxes_and_confs = np.concatenate([all_boxes, max_confidences[:, None], max_conf_classes[:, None]], axis=1)

    #     t0 = time.time()
    #     optim_boxes, optim_scores, optim_classes = nms_optimized(predictions=all_boxes_and_confs,
    #                                                              conf_thres=min(self.classes_conf.values()),
    #                                                              iou_thres=self.iou_thres)
    #     for idx, box in enumerate(optim_boxes):
    #         idx_class = int(optim_classes[idx])
    #         idx_score = optim_scores[idx]
    #         idx_box = [max(0, value) for value in box[:4].tolist()]
    #         if idx_score > self.classes_conf[idx_class]:
    #             self.predictions.append(idx_box + [idx_score, idx_class])

    def _filter_detect_tiny_predictions(self, preds):
        if self.trt_version == 10:
            num_dets, det_boxes, det_scores, det_classes = preds[:4]

            for pred_idx in range(num_dets.tolist()[0][0]):
                idx_class = int(det_classes[0, pred_idx])
                idx_score = det_scores[0, pred_idx]
                idx_box = [max(0, value) for value in det_boxes[0, pred_idx].tolist()]
                if idx_class in self.classes_conf:
                    if idx_score > self.classes_conf[idx_class]:
                        self.predictions.append(idx_box + [idx_score, idx_class])

        elif self.trt_version == 8:
            preds = preds[0]
            print("preds = non_max_suppression_cpu", preds)
            # preds = non_max_suppression_cpu(prediction=preds[0],
            #                                 conf_thres=min(self.classes_conf.values()),
            #                                 iou_thres=self.iou_thres)
            try:
                for idx, box in enumerate(preds):
                    idx_class = int(box[-1])
                    idx_score = box[-2]
                    idx_box = [max(0, value) for value in box[:4].tolist()]
                    if idx_class in self.classes_conf:
                        if idx_score > self.classes_conf[idx_class]:
                            self.predictions.append(idx_box + [idx_score, idx_class])
            except Exception as e:
                print(e)

    def _filter_detect_lstm_predictions(self, preds):
        self.predictions.append(preds[0])

    def infer(self, batch):
        # Copy I/O and Execute
        self._clear_predictions()
        try:
            cuda.memcpy_htod(self.inputs[0]['allocation'], batch)
        except Exception as e:
            print("e: ", e)
             
        self.context.execute_v2(self.allocations)
        for o in range(len(self.outputs)):
            cuda.memcpy_dtoh(self.outputs[o]['host_allocation'], self.outputs[o]['allocation'])

        print("start")
        preds = [o['host_allocation'] for o in self.outputs]
        print("self.infer_type", preds[1])

        # Process predictions based on the inference type
        if self.infer_type == 'detection':
            self._filter_detect_predictions(preds)
        elif self.infer_type == 'segmentation':
            self._filter_segm_predictions(preds)
        elif self.infer_type == 'detection_tiny':
            self._filter_detect_tiny_predictions(preds)
        elif self.infer_type == 'lstm':
            self._filter_detect_lstm_predictions(preds)

        return self.predictions