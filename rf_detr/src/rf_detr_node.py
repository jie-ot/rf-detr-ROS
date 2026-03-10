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

COCO_CLASSES =[
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

class RFDetrTRTNode:
    def __init__(self):
        # 初始化 ROS 节点
        rospy.init_node('rf_detr_trt_node')
        rospy.loginfo("rf-detr节点启动")
        
        # 读取参数
        self.task = rospy.get_param('~task', 'detection')
        self.size = rospy.get_param('~size', 'small')
        self.conf_threshold = rospy.get_param('~conf_threshold', 0.5)
        self.input_topic = rospy.get_param('~input_topic', '/camera/image_raw')
        engine_filename = rospy.get_param('~engine_file', 'rf_detr_small.engine')
        
        # 定位引擎文件路径
        rospack = rospkg.RosPack()
        try:
            package_path = rospack.get_path('rf_detr')
        except rospkg.ResourceNotFound:
            rospy.logerr("Cannot find ROS package 'rf_detr'.")
            rospy.signal_shutdown("Package Not Found")
            return
            
        self.engine_path = os.path.join(package_path, 'models', engine_filename)
        if not os.path.exists(self.engine_path):
            rospy.logerr(f"TensorRT Engine not found at: {self.engine_path}")
            rospy.logerr("Hint: Did you convert the ONNX model to .engine using trtexec?")
            rospy.signal_shutdown("Missing Engine")
            return

        self.bridge = CvBridge()
        
        # 初始化 CUDA 上下文
        cuda.init()
        self.device = cuda.Device(0)
        self.ctx = self.device.make_context()
        
        # 加载模型引擎
        rospy.loginfo(f"Loading TensorRT Engine: {engine_filename}")
        self.logger = trt.Logger(trt.Logger.WARNING) 
        
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        # 为模型分配固定显存
        self._allocate_buffers()
        
        # 定义 RF-DETR 官方预处理使用的 ImageNet 均值和方差
        self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.img_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        rospy.loginfo("[RF-DETR-TRT]模型加载成功！")
        rospy.loginfo(f"   - Task       : {self.task.upper()}")
        rospy.loginfo(f"   - Size       : {self.size.upper()}")
        rospy.loginfo(f"   - Threshold  : {self.conf_threshold}")
        rospy.loginfo(f"   Listening to : {self.input_topic}")
        
        # 发布与订阅
        self.pub = rospy.Publisher('/detections', DetectionArray, queue_size=10)
        self.sub = rospy.Subscriber(self.input_topic, Image, self.image_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

    def _allocate_buffers(self):
        """分配 GPU 显存，如果引擎转为了 FP16，这里的 dtype 会自动设为 float16"""
        self.inputs = []
        self.outputs =[]
        self.bindings =[]
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

    def _sigmoid(self, x):
        """原生 Numpy 版本的 Sigmoid，加入 clip 防止 exp() 溢出报警"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -250, 250)))

    def image_callback(self, msg):
        try:
            self.ctx.push() # 注入 CUDA 上下文
            
            # ==========================================
            # 模块一：符合官方规范的极速前处理
            # ==========================================
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            orig_h, orig_w = cv_image.shape[:2]
            
            # 自动获取引擎期望的输入分辨率 (通常为 640x640)
            input_shape = self.inputs[0]['shape']
            input_h, input_w = input_shape[2], input_shape[3]
            
            # 缩放并转 RGB
            resized = cv2.resize(cv_image, (input_w, input_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # [修正!] 必须遵守 ViT 的标准化: (x/255.0 - mean) / std
            img_norm = (rgb.astype(np.float32) / 255.0 - self.img_mean) / self.img_std
            
            # HWC -> CHW -> BCHW
            img_tensor = img_norm.transpose(2, 0, 1)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            
            # 动态适应 FP16/FP32，拷贝进内存池
            np.copyto(self.inputs[0]['host'], img_tensor.astype(self.inputs[0]['dtype']).ravel())
            
            # ==========================================
            # 模块二：GPU TensorRT 零延迟执行
            # ==========================================
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            self.stream.synchronize()
            
            # ==========================================
            # 模块三：基于官方源码逻辑的后处理
            # ==========================================
            logits_host = None
            boxes_host = None
            masks_host = None
            
            # 动态解析输出端口 (因为分类头、框头和掩码头的顺序在导出时可能不同)
            for out in self.outputs:
                shape = out['shape']
                data = out['host'].reshape(shape)
                if shape[-1] == 4:
                    boxes_host = data[0]                     # 提取 Bounding boxes [300, 4]
                elif shape[-1] == len(COCO_CLASSES):
                    logits_host = data[0]                    # 提取 原始 Logits[300, 80]
                elif len(shape) >= 3:
                    masks_host = data[0]                     # 提取 分割掩码 [300, H, W]
            
            if logits_host is None or boxes_host is None:
                self.ctx.pop()
                return

            out_msg = DetectionArray()
            out_msg.header = msg.header # 继承原图时间戳和帧ID
            
            # [修正!] ONNX 输出的是 logits，必须经过 sigmoid 转换成概率 (0~1) 才能应用阈值！
            scores = self._sigmoid(logits_host)
            
            max_scores = np.max(scores, axis=1)
            class_ids = np.argmax(scores, axis=1)
            
            # 大于置信度阈值的对象才会被留下
            valid_mask = max_scores > self.conf_threshold
            
            valid_scores = max_scores[valid_mask]
            valid_class_ids = class_ids[valid_mask]
            valid_boxes = boxes_host[valid_mask]
            
            if masks_host is not None and self.task == "segmentation":
                valid_masks_logits = masks_host[valid_mask]
            else:
                valid_masks_logits = None
            
            # 遍历打包
            for i in range(len(valid_scores)):
                det = Detection()
                det.score = float(valid_scores[i])
                det.class_id = int(valid_class_ids[i])
                det.class_name = COCO_CLASSES[det.class_id] if det.class_id < len(COCO_CLASSES) else "Unknown"
                
                # DETR 框还原机制: (cx, cy, w, h) -> 原图像素坐标 (x1, y1, x2, y2)
                cx, cy, w, h = valid_boxes[i]
                x1 = (cx - w / 2) * orig_w
                y1 = (cy - h / 2) * orig_h
                x2 = (cx + w / 2) * orig_w
                y2 = (cy + h / 2) * orig_h
                
                det.bbox =[
                    max(0, float(x1)), 
                    max(0, float(y1)), 
                    min(orig_w, float(x2)), 
                    min(orig_h, float(y2))
                ]
                
                # 分割掩码 (Segmentation Mask) 还原处理
                if valid_masks_logits is not None:
                    # [修正!] 分割出的 mask 也是 logits，同样需要激活和阈值化处理
                    mask_prob = self._sigmoid(valid_masks_logits[i])
                    # 生成二值化掩码图 (大于0.5即为前景)并扩大到0-255方便图像显示
                    mask_uint8 = (mask_prob > 0.5).astype(np.uint8) * 255
                    
                    # 重新拉伸回相机的原始分辨率 (原位插值)
                    mask_resized = cv2.resize(mask_uint8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    det.mask = self.bridge.cv2_to_imgmsg(mask_resized, encoding="mono8")
                
                out_msg.detections.append(det)
                
            self.pub.publish(out_msg)
            
        except Exception as e:
            rospy.logerr_throttle(2.0, f"[RF-DETR] Error in callback: {e}")
        finally:
            self.ctx.pop()

if __name__ == '__main__':
    try:
        node = RFDetrTRTNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass   
