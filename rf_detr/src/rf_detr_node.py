"""
版本1：适用于.onnx文件，如果无法用TensorRT优化，就用这个版本
"""
#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import os
import rospkg
import onnxruntime as ort
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
 
class RFDetrORTNode:
    def __init__(self):
        rospy.init_node('rf_detr_ort_node')
        rospy.on_shutdown(self.cleanup)
        
        # 参数
        self.conf_threshold = rospy.get_param('~conf_threshold', 0.5)
        self.input_h = rospy.get_param('~input_height', 672)
        self.input_w = rospy.get_param('~input_width', 672)
        model_name = rospy.get_param('~model_file', 'infer_small_model.sim.onnx')
        
        # 路径
        rospack = rospkg.RosPack()
        model_path = os.path.join(rospack.get_path('rf_detr'), 'models', model_name)
        
        # 加载 ONNX Session (指定使用 CUDA)
        rospy.loginfo(f"Loading ONNX model: {model_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        self.bridge = CvBridge()
        self.pub = rospy.Publisher('/detections', DetectionArray, queue_size=10)
        self.sub = rospy.Subscriber(rospy.get_param('~input_topic', '/camera/image_raw'), Image, self.image_callback, queue_size=1, tcp_nodelay=True)
        rospy.loginfo("[RF-DETR] Node started successfully.")

    def preprocess(self, img):
        orig_h, orig_w = img.shape[:2]
        # Letterbox 缩放和 Resize
        ratio = min(self.input_w / orig_w, self.input_h / orig_h)
        new_unpad = (int(orig_w * ratio), int(orig_h * ratio))
        img_resized = cv2.resize(img, new_unpad)
        
        # 填充
        dw, dh = (self.input_w - new_unpad[0]) // 2, (self.input_h - new_unpad[1]) // 2
        canvas = np.full((self.input_h, self.input_w, 3), 128, dtype=np.uint8)  # 填充灰色
        canvas[dh:dh+new_unpad[1], dw:dw+new_unpad[0]] = img_resized
        
        # 标准化
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_rgb.astype(np.float32) / 255.0 - mean) / std
        input_tensor = np.expand_dims(img_norm.transpose(2, 0, 1), axis=0).astype(np.float32)
        
        return input_tensor, ratio, dw, dh, orig_h, orig_w

    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            input_tensor, ratio, dw, dh, orig_h, orig_w = self.preprocess(img)
            
            # 推理
            outputs = self.session.run(None, {"input": input_tensor})
            boxes, logits = outputs[0][0], outputs[1][0]
            
            # Sigmoid & 筛选
            scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -250, 250)))
            max_scores = np.max(scores, axis=1)
            class_ids = np.argmax(scores, axis=1)
            
            # 过滤: 高于阈值 且 不是背景(0)
            mask = (max_scores > self.conf_threshold) & (class_ids != 0)
            
            valid_boxes = boxes[mask]
            valid_scores = max_scores[mask]
            valid_classes = class_ids[mask] - 1 
            
            # 组装消息
            out_msg = DetectionArray()
            out_msg.header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id) 
            
            for i in range(len(valid_scores)):
                cx, cy, w, h = valid_boxes[i]
                
                # 坐标反算到原图
                cx_p = cx * self.input_w
                cy_p = cy * self.input_h
                w_p = w * self.input_w
                h_p = h * self.input_h
                
                x1 = (cx_p - w_p / 2 - dw) / ratio
                y1 = (cy_p - h_p / 2 - dh) / ratio
                x2 = (cx_p + w_p / 2 - dw) / ratio
                y2 = (cy_p + h_p / 2 - dh) / ratio
                
                det = Detection()
                det.score = float(valid_scores[i])
                det.class_id = int(valid_classes[i])
                # 增加越界保护
                det.class_name = COCO_CLASSES[det.class_id] if 0 <= det.class_id < len(COCO_CLASSES) else "Unknown"
                det.bbox = [max(0, float(x1)), max(0, float(y1)), min(orig_w, float(x2)), min(orig_h, float(y2))]
                out_msg.detections.append(det)
                
            self.pub.publish(out_msg)
        except Exception as e:
            rospy.logerr(f"[RF-DETR] Callback error: {e}")

    def cleanup(self):
        rospy.loginfo("Cleanup complete.")

if __name__ == '__main__':
    RFDetrORTNode()
    rospy.spin()
