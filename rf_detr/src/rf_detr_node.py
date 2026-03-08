#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import os
import rospkg
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rf_detr.msg import Detection, DetectionArray
from PIL import Image as PILImage
from rfdetr.util.coco_classes import COCO_CLASSES

class RFDetrNode:
    def __init__(self):
        rospy.init_node('rf_detr_node')
        
        # 节点启动的打印信息
        rospy.loginfo("==================================================")
        rospy.loginfo("🚀 [RF-DETR] Detector Node is starting...")
        rospy.loginfo("==================================================")
        
        # 1. 读取参数
        self.task = rospy.get_param('~task', 'detection')
        self.size = rospy.get_param('~size', 'small')
        self.conf_threshold = rospy.get_param('~conf_threshold', 0.5)
        self.input_topic = rospy.get_param('~input_topic', '/camera/image_raw')
        checkpoint_file = rospy.get_param('~checkpoint', 'rf-detr-small.pth')
        
        # 2. 定位模型路径
        rospack = rospkg.RosPack()
        try:
            package_path = rospack.get_path('rf_detr')
        except rospkg.ResourceNotFound:
            rospy.logerr("❌ [RF-DETR] Cannot find ROS package 'rf_detr'.")
            rospy.signal_shutdown("Package Not Found")
            return
            
        if checkpoint_file != "":
            self.model_path = os.path.join(package_path, 'models', checkpoint_file)
            if not os.path.exists(self.model_path):
                rospy.logerr(f"❌ [RF-DETR] Checkpoint not found at: {self.model_path}")
                rospy.signal_shutdown("Missing checkpoint file")
                return
        else:
            self.model_path = None
            
        self.bridge = CvBridge()
        
        # 3. 加载模型
        rospy.loginfo(f"⏳ [RF-DETR] Loading model[Task: {self.task} | Size: {self.size}]...")
        self.model = self._load_model(self.task, self.size, self.model_path)
        
        if self.model is None:
            rospy.signal_shutdown("Failed to load model")
            return
            
        # 成功加载模型的打印信息
        rospy.loginfo("✅ [RF-DETR] Model loaded successfully!")
        rospy.loginfo(f"   - Task       : {self.task.upper()}")
        rospy.loginfo(f"   - Size       : {self.size.upper()}")
        rospy.loginfo(f"   - Checkpoint : {checkpoint_file if checkpoint_file else 'Auto-downloaded'}")
        rospy.loginfo(f"   - Threshold  : {self.conf_threshold}")
        rospy.loginfo(f"📡 [RF-DETR] Listening to topic: {self.input_topic}")
        rospy.loginfo("==================================================")
        
        # 4. 初始化发布者和订阅者
        self.pub = rospy.Publisher('/rf_detr/detections', DetectionArray, queue_size=1)
        self.sub = rospy.Subscriber(self.input_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        
    def _load_model(self, task, size, model_path):
        kwargs = {}
        if model_path:
            kwargs['checkpoint'] = model_path
            
        try:
            if task == "detection":
                if size == "nano":
                    from rfdetr import RFDETRNano; return RFDETRNano(**kwargs)
                elif size == "small":
                    from rfdetr import RFDETRSmall; return RFDETRSmall(**kwargs)
                elif size == "medium":
                    from rfdetr import RFDETRMedium; return RFDETRMedium(**kwargs)
            elif task == "segmentation":
                if size == "nano":
                    from rfdetr import RFDETRSegNano; return RFDETRSegNano(**kwargs)
                elif size == "small":
                    from rfdetr import RFDETRSegSmall; return RFDETRSegSmall(**kwargs)
        except Exception as e:
            rospy.logerr(f"❌ [RF-DETR] Failed to instantiate model: {e}")
            
        return None

    def image_callback(self, msg):
        try:
            # 1. 图像格式转换: ROS Image -> OpenCV (RGB) -> PIL Image (保持原始分辨率)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            pil_image = PILImage.fromarray(cv_image)
            
            # 2. 执行推理
            detections_sv = self.model.predict(pil_image, threshold=self.conf_threshold)
            
            # 3. 初始化输出的 ROS 消息
            out_msg = DetectionArray()
            out_msg.header = msg.header
            
            # 解析 supervision 结果
            xyxy = detections_sv.xyxy
            confidences = detections_sv.confidence
            class_ids = detections_sv.class_id
            masks = detections_sv.mask
            
            # 4. 打包检测结果
            if xyxy is not None and len(xyxy) > 0:
                for i in range(len(xyxy)):
                    det = Detection()
                    
                    # 取出 bounding box (直接使用原图坐标)
                    det.bbox = xyxy[i].tolist()
                    det.score = float(confidences[i])
                    det.class_id = int(class_ids[i])
                    
                    if det.class_id < len(COCO_CLASSES):
                        det.class_name = COCO_CLASSES[det.class_id]
                    else:
                        det.class_name = str(det.class_id)
                    
                    # 赋值分割 Mask
                    if masks is not None:
                        mask_uint8 = (masks[i] * 255).astype(np.uint8)
                        det.mask = self.bridge.cv2_to_imgmsg(mask_uint8, encoding="mono8")
                    
                    out_msg.detections.append(det)
                    
            self.pub.publish(out_msg)
            
        except Exception as e:
            rospy.logerr_throttle(2.0, f"❌ [RF-DETR] Error in callback: {e}")

if __name__ == '__main__':
    try:
        node = RFDetrNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
