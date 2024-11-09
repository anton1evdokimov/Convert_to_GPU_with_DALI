from typing import Tuple
import cupy as cp
import numpy as np
import cvcuda

class ImgPipeline:
    
    def __init__(self, img_size: int):
        self.img_size = img_size
        return None
    
    def process(self, img: np.ndarray) -> np.ndarray:
        self._put_img(img)
        self._resize_()
        self._pad_()
        self._bgr_hwc2rgb_chw_()
        self.img = self.img[None, ...]
        # self.img = self.img.astype(np.float32) / 255.
        self.img = cvcuda.convertto(self.img, np.float32, scale=1/255.)
        
        return self.img
    
    def _put_img(self, img: np.ndarray) -> None:
        self.img = img
        self.h0, self.w0 = self.img.shape[:2]
        self._define_scaling_parameters()
        return None
    
    def _define_scaling_parameters(self)-> None:
        # resize down
        self.scale = cp.max(cp.array([self.h0 / self.img_size, self.w0 / self.img_size]))
        if self.scale > 1:
            self.h, self.w = (np.round(self.h0 / self.scale)), int(np.round(self.w0 / self.scale))
        else:
            self.h, self.w = self.h0, self.w0
        
        #padding
        dh, dw = abs(self.img_size - self.h), abs(self.img_size - self.w)  # wh padding
        dh, dw = dh / 2, dw /2  # divide padding into 2 sides
        self.top, self.bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        self.left, self.right = int(round(dw - 0.1)), int(round(dw + 0.1))
        return None
    
    def _pad_(self, padding_value: int=114) -> None:
        self.img = np.pad(self.img, ((self.top, self.bottom), (self.left, self.right), (0, 0)), mode='constant', constant_values=padding_value)
        return None
    
    def _resize_(self)-> None:
        if self.scale > 1:
            self.img = cvcuda.resize(self.img, (self.w, self.h), interpolation=cvcuda.Interp.AREA)
        return None
    
    def _bgr_hwc2rgb_chw_(self) -> None:
        self.img = self.img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, to (C, H, W)
        return None
    
    def coords_clip(self, coords: np.ndarray) -> np.ndarray:
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        coords[:, 0].clip(0, self.w0, out=coords[:, 0])  # x1
        coords[:, 1].clip(0, self.h0, out=coords[:, 1])  # y1
        coords[:, 2].clip(0, self.w0, out=coords[:, 2])  # x2
        coords[:, 3].clip(0, self.h0, out=coords[:, 3])  # y2
        return coords

    def coords_unscale(self, coords: np.ndarray) -> np.ndarray:
        # xy xy shift; dhdw_shift
        coords -= np.array([self.left, self.top, self.left, self.top], dtype=coords.dtype)
        if self.scale > 1:
            coords *= self.scale #unscale coordinates 
        coords = self.coords_clip(coords.round())
        return coords
    
    @staticmethod
    def data_decode(data: bytes)-> Tuple[str, np.ndarray]:
        frame_id, img_bytes = data.split(b';', 1)
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        return frame_id.decode(), img