"""
版本3：增加了TensorRT加速的版本
"""
#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import os
import rospkg
import tensorrt as trt
import pycuda.driver as cuda
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from rf_detr.msg import Detection, DetectionArray

# COCO 类别列表
COCO_CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

class RFDetrTRTNode:
    def __init__(self):
        rospy.init_node('rf_detr_trt_node')
        rospy.on_shutdown(self.cleanup)
        
        # 参数
        self.conf_threshold = rospy.get_param('~conf_threshold', 0.5)
        self.input_h = rospy.get_param('~input_height', 672)
        self.input_w = rospy.get_param('~input_width', 672)
        engine_name = rospy.get_param('~engine_file', 'rf_detr_small.engine')
        
        # 路径
        rospack = rospkg.RosPack()
        engine_path = os.path.join(rospack.get_path('rf_detr'), 'models', engine_name)
        
        # 1. TensorRT 初始化
        cuda.init()
        self.device = cuda.Device(0)
        self.ctx = self.device.make_context()
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        rospy.loginfo(f"Loading TensorRT Engine: {engine_path}")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        self._allocate_buffers()
        
        self.bridge = CvBridge()
        self.pub = rospy.Publisher('/detections', DetectionArray, queue_size=10)
        self.sub = rospy.Subscriber(rospy.get_param('~input_topic', '/camera/image_raw'), 
                                    Image, self.image_callback, queue_size=1, tcp_nodelay=True)
        rospy.loginfo("[RF-DETR] TensorRT Node started successfully.")

    def _allocate_buffers(self):
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': self.engine.get_binding_shape(binding), 'dtype': dtype})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': self.engine.get_binding_shape(binding), 'dtype': dtype})

    def preprocess(self, img):
        orig_h, orig_w = img.shape[:2]
        ratio = min(self.input_w / orig_w, self.input_h / orig_h)
        new_unpad = (int(orig_w * ratio), int(orig_h * ratio))
        img_resized = cv2.resize(img, new_unpad)
        
        dw, dh = (self.input_w - new_unpad[0]) // 2, (self.input_h - new_unpad[1]) // 2
        canvas = np.full((self.input_h, self.input_w, 3), 128, dtype=np.uint8)
        canvas[dh:dh+new_unpad[1], dw:dw+new_unpad[0]] = img_resized
        
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_rgb.astype(np.float32) / 255.0 - mean) / std
        return img_norm.transpose(2, 0, 1).ravel(), ratio, dw, dh, orig_h, orig_w

    def image_callback(self, msg):
        try:
            self.ctx.push() # 关键：进入 CUDA 上下文
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            input_flat, ratio, dw, dh, orig_h, orig_w = self.preprocess(img)
            
            # 推理
            np.copyto(self.inputs[0]['host'], input_flat)
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            self.stream.synchronize()
            
            # 后处理
            boxes = self.outputs[0]['host'].reshape(self.outputs[0]['shape'])[0]
            logits = self.outputs[1]['host'].reshape(self.outputs[1]['shape'])[0]
            
            scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -250, 250)))
            max_scores = np.max(scores, axis=1)
            class_ids = np.argmax(scores, axis=1)
            mask = (max_scores > self.conf_threshold) & (class_ids != 0)
            
            out_msg = DetectionArray()
            out_msg.header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
            
            for i, box in enumerate(boxes[mask]):
                cx, cy, w, h = box
                x1 = (cx * self.input_w - (w * self.input_w) / 2 - dw) / ratio
                y1 = (cy * self.input_h - (h * self.input_h) / 2 - dh) / ratio
                x2 = (cx * self.input_w + (w * self.input_w) / 2 - dw) / ratio
                y2 = (cy * self.input_h + (h * self.input_h) / 2 - dh) / ratio
                
                det = Detection()
                det.score = float(max_scores[mask][i])
                det.class_id = int(class_ids[mask][i] - 1)
                det.class_name = COCO_CLASSES[det.class_id] if 0 <= det.class_id < len(COCO_CLASSES) else "Unknown"
                det.bbox = [max(0, float(x1)), max(0, float(y1)), min(orig_w, float(x2)), min(orig_h, float(y2))]
                out_msg.detections.append(det)
                
            self.pub.publish(out_msg)
        except Exception as e:
            rospy.logerr(f"[RF-DETR] Callback error: {e}")
        finally:
            self.ctx.pop()

    def cleanup(self):
        if hasattr(self, 'ctx'): self.ctx.pop()
        rospy.loginfo("Resources cleaned.")

if __name__ == '__main__':
    try:
        node = RFDetrTRTNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
