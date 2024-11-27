from typing import Union, List
import sys

from inference.model_trt_cvcuda import ObjectDetectionTensorRT
from inference.model_trt_prod import TensorRTInference
from inference.model_trt import ModelTRT

sys.path.insert(0, '..')

import pycuda.driver as cuda
import numpy as np
import cvcuda # cvcuda_cu12-0.7.0b0-cp310-cp310-linux_x86_64.whl
from nvidia import nvimgcodec # pip install nvidia-nvimgcodec-cu12
import torch

import numba
import cv2

# from model_trt import ModelTRT

# print("cvcuda.__version__", cvcuda.__version__)

class ImgPipeliner:
    
    TYPE_DETECTION = 0
    TYPE_CLASSIFICATON = 1
    
    MEAN = [0.485, 0.456, 0.406] 
    STD = [0.229, 0.224, 0.225]
    
    
    def __init__(self, img_size: int, type_process: int, padding_value: int = 0):
        self.decoder = nvimgcodec.Decoder(
            backends=[nvimgcodec.Backend(nvimgcodec.GPU_ONLY, load_hint=0.5), nvimgcodec.Backend(nvimgcodec.HYBRID_CPU_GPU)])
        self.params_dec = nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.RGB) # unnecessary, rgb is used as default already. This line is just for clearness
        self.img_size = img_size
        self.type_process = type_process
        self.padding_value = padding_value
        #   test preallocating tensor in order to not create a new one for each function call
        # assert 
        # help('cvcuda.convertto')
        # print("cvcuda", dir(cvcuda))
        if self.type_process == self.TYPE_CLASSIFICATON:
            self.mean = np.array(self.MEAN, dtype=np.float32).reshape(1, 1, 3)
            self.mean = nvimgcodec.as_image(self.mean)
            self.mean = cvcuda.as_tensor(self.mean.cuda(), "HWC")

            self.std = np.array(self.STD, dtype=np.float32).reshape(1, 1, 3)
            self.std = nvimgcodec.as_image(self.std)
            self.std = cvcuda.as_tensor(self.std.cuda(), "HWC")
        return None
    
    
    def decode(self, img: Union[np.ndarray,  bytes]) -> nvimgcodec.Image:
        img = self.decoder.decode(img, params=self.params_dec)
        return img
    

    def process(self, img: Union[np.ndarray,  bytes]) -> cvcuda.Tensor:
        img = self.decode(img) # default RGB
        
        img = cvcuda.as_tensor(img, "HWC")
       
        self.h0, self.w0, _ = img.shape
        
        if self.type_process == self.TYPE_DETECTION:
            img = self._process_detection(img, self.padding_value)
        elif self.type_process == self.TYPE_CLASSIFICATON:
            img = self._process_classification(img)
        img = cvcuda.reformat(img, 'CHW')
        img = cvcuda.stack([img]) # CHW -> NCHW
        return img
    
    def _process_detection(self, img, padding_value: int):
        self._define_scaling_parameters()
        if self.scale > 1:
            img = cvcuda.resize(img, (self.h, self.w, 3), cvcuda.Interp.LINEAR)
        img = cvcuda.copymakeborder(img,
            border_value=(padding_value, padding_value, padding_value),
            top=self.top,
            bottom=self.bottom,
            left=self.left,
            right=self.right)
        img = cvcuda.convertto(img, np.float32, scale=1/255.)
        return img
    
    def _process_classification(self, img):
        img = cvcuda.resize(img, (self.img_size, self.img_size, 3), cvcuda.Interp.AREA)
        img = cvcuda.convertto(img, np.float32, scale=1/255.)
        img = cvcuda.normalize(img, base=self.mean, scale=self.std, flags=cvcuda.NormalizeFlags.SCALE_IS_STDDEV)
        return img
    
    def _define_scaling_parameters(self)-> None:
        # resize down
        self.scale = np.max([self.h0 / self.img_size, self.w0 / self.img_size])
        if self.scale > 1:
            self.h, self.w = int(np.round(self.h0 / self.scale)), int(np.round(self.w0 / self.scale))
            
        else:
            self.h, self.w = self.h0, self.w0
        
        #padding
        dh, dw = abs(self.img_size - self.h), abs(self.img_size - self.w)  # wh padding
        
        dh, dw = dh / 2, dw /2  # divide padding into 2 sides
        self.top, self.bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        self.top, self.bottom = 12, 12
        self.left, self.right = int(round(dw - 0.1)), int(round(dw + 0.1))
        return None
    
    def coords_unscale(self, xyxy: np.ndarray) -> np.ndarray:
        # xy xy shift; dhdw_shift
        xyxy -= np.array([self.left, self.top, self.left, self.top], dtype=xyxy.dtype)
        if self.scale > 1:
            xyxy *= self.scale # unscale coordinates 
        xyxy = self.coords_clip(xyxy.round())
        return xyxy
    
    def coords_clip(self, xyxy: np.ndarray) -> np.ndarray:
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        xyxy[:, 0].clip(0, self.w0, out=xyxy[:, 0])  # x1
        xyxy[:, 1].clip(0, self.h0, out=xyxy[:, 1])  # y1
        xyxy[:, 2].clip(0, self.w0, out=xyxy[:, 2])  # x2
        xyxy[:, 3].clip(0, self.h0, out=xyxy[:, 3])  # y2
        return xyxy

# need to explore more
def cuda_init(device_id=0):
    # Define the cuda device, context
    cuda_device = cuda.Device(device_id)
    cuda_ctx = cuda_device.retain_primary_context()
    cuda_ctx.push()
    # even if exception occurs somewhere, we nethertheless must call cuda_ctx.pop()
    return cuda_ctx

class CudaContext:
    
    def __init__(self, device_id=0, logger=None) -> None:
        self.device_id = device_id
        self.logger = logger
        return None
    
    def __enter__(self):
        cuda_device = cuda.Device(self.device_id)
        self.cuda_ctx = cuda_device.retain_primary_context()
        self.cuda_ctx.push()
        print("CudaContext",  self.cuda_ctx)
        
        return None
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cuda_ctx.pop()
        print("exc_type", exc_tb)
        # when true, any exception happened inside the "with block" is suppressed
        return True

def test_pipeline_detection(img: bytes):
    import cupy as cp

    
    img_size = 640
    pipeliner = ImgPipeliner(img_size=img_size, type_process=ImgPipeliner.TYPE_DETECTION)
    img = pipeliner.process(img)

    img = cp.asnumpy(cp.asarray(img.cuda()))
    img = img.reshape(3, img_size, img_size).transpose(1,2,0)
    img = (img * 255).astype(np.uint8)
    cv2.imwrite('lol.jpg', img[:, :, ::-1])
    return None

def test_pipeline_classification(img: bytes):
    import cv2
    import cupy as cp
    
    img_size = 384
    pipeliner = ImgPipeliner(img_size=img_size, type_process=ImgPipeliner.TYPE_CLASSIFICATON)
    img = pipeliner.process(img)
    
    img = cp.asnumpy(cp.asarray(img.cuda()))
    img = img.reshape(3, img_size, img_size).transpose(1,2,0)
    mean = np.array(ImgPipeliner.MEAN).reshape(1, 1, 3)
    std = np.array(ImgPipeliner.STD).reshape(1, 1, 3)
    img = (img * std) + mean
    img = (img * 255).astype(np.uint8)
    cv2.imwrite('lol.jpg', img[:, :, ::-1])
    return None

def plot_bboxes_old(img, bboxes: np.ndarray, scores: np.ndarray, labels: Union[List[str], np.ndarray], color=(255, 0, 0)):
    
    for bbox, score, label in zip(bboxes, scores, labels):
        if score < 0.5:
            continue
        x0 = int(bbox[0])
        y0 = int(bbox[1])
        x1 = int(bbox[2])
        y1 = int(bbox[3])
        if not isinstance(label, str):
            label = str(label.item())
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        
        text = f'class:{label}, score:{score:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 1, 1)[0]
        cv2.putText(img, text, (x0, y0 - txt_size[1]), font, 0.8, (0, 0, 0), thickness=2)
    return None

def plot_bboxes(img, bboxes: torch.Tensor, color=(255, 0, 0)):
    
    for *bbox, score, label in bboxes:
        if score < 0.5:
            continue
        x0 = int(bbox[0])
        y0 = int(bbox[1])
        x1 = int(bbox[2])
        y1 = int(bbox[3])
        if not isinstance(label, str):
            label = str(label.item())
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        
        text = f'score:{score:.2f}, class:{label}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 1, 1)[0]
        cv2.putText(img, text, (x0 - 20, y0 - txt_size[1]), font, 0.8, (0, 0, 0), thickness=2)
    return None

weights_path = "../weights/yolov7_18.11.2024.trt"
model_params = {
    'weights': weights_path,
    'classes_ids_conf': [0.5 for i in range(10)],
    'infer_type': 'detection_tiny',
    'iou_thres': 0.5,
}
def clip_coords_old(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def scale_coords_old(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img0_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[0].clamp_(0, img0_shape[1])  # x1
    boxes[1].clamp_(0, img0_shape[0])  # y1
    boxes[2].clamp_(0, img0_shape[1])  # x2
    boxes[3].clamp_(0, img0_shape[0])  # y2

def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    
    coords[[0, 2]] -= pad[0]  # x padding
    coords[[1, 3]] -= pad[1]  # y padding
    coords[:4] /= gain
    
    coords[0].clamp_(0, img0_shape[1])  # x1
    coords[1].clamp_(0, img0_shape[0])  # y1
    coords[2].clamp_(0, img0_shape[1])  # x2
    coords[3].clamp_(0, img0_shape[0])  # y2
    # clip_coords(coords, img0_shape)
   
    return coords

img_path = './inference/source/5.jpg'

import cupy as cp
import glob
import os
def test_gpu_gpu_inference(imgCV):
    # cvcuda_stream = cvcuda.Stream().current
    
    # with open(weights_path, 'rb') as file:
    #     trt_weights = file.read()
        
    source_shape = torch.tensor(imgCV.shape).cuda()
    inference_shape = torch.tensor([384, 672])
    img_size = 640
    
    with CudaContext():
        try:
            model = ObjectDetectionTensorRT()
            pipeliner = ImgPipeliner(img_size=img_size, type_process=ImgPipeliner.TYPE_DETECTION)
            
            for img_path in glob.glob("./inference/source/*.jpg"):
                with open(img_path, 'rb') as file:
                    img0 = file.read()
                imgCV = cv2.imread(img_path)
                img = pipeliner.process(img0)
                
                # b = torch.as_tensor(img.cuda())  # CPU
                # b = b.squeeze().cpu() #.numpy().transpose(1,2,0)
                # b = np.array(b.permute(1, 2, 0)*255, dtype=np.uint8)
                # b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
                preds = model(img)
                for i, pred in enumerate(preds):
                    if pred[4] == 0:
                        break
                    preds[i][:4]=scale_coords(inference_shape, pred[:4], source_shape)
                # pipeliner.coords_unscale(bboxes[0])
                # plot_bboxes(imgCV, preds)
                saved_path = os.path.join('./inference/output_after_preprocessing', img_path.split("/")[-1])
                print("saved_path", saved_path)
                cv2.imwrite(saved_path, b)
                
        except Exception as e:
                print("Exception: ", e)
    
    return None

def test_gpu_tensorrt_load_inference(img0: bytes, imgCV, num_runs = 1000):
    import time
    
    # with open('./weights/yolov7_18.11.2024.trt', 'rb') as file:
    #     weights = file.read()

    with CudaContext():
        try:
            img_size = 640
            pipeliner = ImgPipeliner(img_size=img_size, type_process=ImgPipeliner.TYPE_DETECTION)
            model = ObjectDetectionTensorRT()
            
            source_shape = cp.array(imgCV.shape) #.cuda()
            inference_shape = cp.array([384, 672])
            t0 = time.perf_counter()
            for _ in range(num_runs):
                img = pipeliner.process(img0)
                preds = model(img)
                for i, pred in enumerate(preds):
                    if pred[4] == 0:
                        break
                    # preds[i][:4]=scale_coords(inference_shape, pred[:4], source_shape)
                
            delta = (time.perf_counter() - t0)
            fps = num_runs / delta
            print(f'One: {delta/(num_runs/1000):.3f} ms')
            print(f'FPS with GPU and TensorRT: {fps:.3f}')
        except Exception as e:
            print("Exception: ", e)
            
    
    return None

def test_cpu_pytorch_load_inference(img0: bytes, img_path: str, num_runs = 1000):
    import cv2
    import time
    from ultralytics import YOLO
    from dataloader.preprocessing import ImgPipeline
    
    # with open('./weights/yolov7_GB_rtx3060.trt', 'rb') as file:
    #     weights = file.read()
    
    # with CudaContext():
    img_size = 640
    pipeliner = ImgPipeline(img_size=img_size)
    # model = ModelTRT(weights, 10, None, True, img_buffer_kind_gpu=False)
    model = YOLO("./best_yolo_11.pt", verbose=False)
    img = cv2.imread(img_path)
    
    t0 = time.perf_counter()
    for _ in range(num_runs):
        # img = cv2.imdecode(np.frombuffer(img0, np.uint8), cv2.IMREAD_COLOR)
        test_img = pipeliner.process(img)[0].transpose(1,2,0)
        res = model(test_img, verbose=False)
    fps = num_runs / (time.perf_counter() - t0)
    print(f'FPS with CPU and Pytorch:{fps:.3f}')
    
    return None


def main():
    import os
    # os.system("dpkg -l | grep nvinfer")# - TenorRT version
    # os.system("trtexec --onnx=/workspace/src/yolo7_10.30_conf_0.01.onnx --workspace=4096 --fp16 --saveEngine=/workspace/weights/yolov7_18.11.2024.trt --verbose --tacticSources=-cublasLt,+cublas")
    
    with open(img_path, 'rb') as file:
        img = file.read()
    
    imgCV=cv2.imread(img_path)
    # test_gpu_gpu_inference(imgCV)
    # test_pipeline_detection(img)
    # test_pipeline_classification(img)
    
    test_gpu_tensorrt_load_inference(img, imgCV)
    # test_cpu_pytorch_load_inference(img, img_path)
    
    # os.system("pip show cvcuda")
    return None

if __name__ == "__main__":
    main()
    


# # gpuarray for some reason does not support float32 -> float16
# array = GPUArray(img.shape, np.float32) 
# cuda.memcpy_dtod_async(array.ptr, img.cuda().__cuda_array_interface__['data'][0], size=array.nbytes, stream=cuda.Stream())
# array = array.astype('float16')