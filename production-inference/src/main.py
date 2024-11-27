from dataloader.preprocessing import ImgPipeline
from inference.models import ModelDetect, ModelTracking
from analytics.bbox import Frame
# from kafka_communication.consumer import KafkaMsgConsumer
# from kafka_communication.producer import KafkaMsgProducer
# from kafka_communication.communication import Orchestrator 
from inference.wghts_ncrptn import Guardian
from environment import CFGGlobal, CFGModel
# from log.logger import get_logger
import glob
import cv2
import os
import cupy as cp
import cvcuda
from nvidia import nvimgcodec
# from numba import cuda
import numpy as np

# logger = get_logger(__name__)
# device = cuda.get_current_device()
# tpb = device.WARP_SIZE       #blocksize или количество потоков на блок, стандартное значение = 32
bpg = 128  # блоков на грид 
    
def _main():
    cfg = CFGGlobal()
    # os.system("trtexec --onnx=/workspace/onnx/best_exp33_14.11.onnx --workspace=4096 --fp16 --saveEngine=/workspace/weights/yolov7_GB_rtx3060.trt --verbose --tacticSources=-cublasLt,+cublas")
    # 
    model        = np.array([1])
    model        = init_model(cfg.MODEL)
    # pipeline     = ImgPipeline(cfg.MODEL.IMG_SIZE)
    # model_detect = ModelDetect(model, pipeline)
    print(model)
    # print(test_cuda[1, tpb](cuda.to_device(model)))
    print(model)
    
    # consumer     = KafkaMsgConsumer(cfg.KAFKA)
    # producer     = KafkaMsgProducer(cfg.KAFKA)
    # orchestrator = Orchestrator(consumer, producer)
    
    # logger.info('Starting inference')
    # for msg in orchestrator:
   
    
     
    for index, img_name in enumerate(glob.glob("/workspace/source/*.jpg")):
    # for index, img_name in enumerate(glob.glob("src/source/*.jpg")):
        # frame = Frame(img_name, kafka_key=img_name)
        img=cv2.imread(img_name)
        print(cvcuda.as_tensor(nvimgcodec.as_image(img).cuda(), "HWC"))
        
        # img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(img_name)
        # preds = model_detect(img)
        # bboxes = ModelDetect.preds2bboxes(*preds, cfg.MODEL)
        # frame.add_bboxes(cp.array(bboxes))
        
        # print("preds", preds)
        
        # orchestrator.send_result(frame)
    # logger.info('Inference has finished')
    return None

def init_model(cfg: CFGModel):
    # weights = cfg.WEIGHTS
    # guardian = Guardian(weights)
    # weights = guardian.decode()
    with open(cfg.WEIGHTS, 'rb') as f:
        weights = f.read()
    try: 
        from inference.model_trt_od import ModelTRT
        model = ModelTRT(weights, cfg.NUM_CLASSES, cfg.DESIRED_CLASSES, is_end2end=cfg.IS_END2END)
    except ModuleNotFoundError: 
        from inference.model_onnx import ModelONNX
        model = ModelONNX(weights, cfg.NUM_CLASSES, cfg.DESIRED_CLASSES)
    return model

def main():
    try:
        _main()
    except Exception as e:
        print(e)
        # logger.critical(e, exc_info=True)    
    return None

if __name__ == '__main__':
    main()