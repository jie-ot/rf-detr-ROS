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
        
        # 1. 精确读取 yaml 配置文件中的参数（严格对应你的命名）
        self.task = rospy.get_param('~task', 'detection')
        self.size = rospy.get_param('~size', 'small')
        self.conf_threshold = rospy.get_param('~conf_threshold', 0.5)
        self.input_topic = rospy.get_param('~input_topic', '/camera/image_raw')
        checkpoint_file = rospy.get_param('~checkpoint', 'rf-detr-small.pth')
        
        # 2. 自动定位 ROS 包路径，并拼接模型的绝对路径
        rospack = rospkg.RosPack()
        try:
            package_path = rospack.get_path('rf_detr')
        except rospkg.ResourceNotFound:
            rospy.logerr("Cannot find ROS package 'rf_detr'. Please source devel/setup.bash!")
            rospy.signal_shutdown("Package Not Found")
            return
            
        if checkpoint_file != "":
            self.model_path = os.path.join(package_path, 'models', checkpoint_file)
            if not os.path.exists(self.model_path):
                rospy.logerr(f"Checkpoint not found at: {self.model_path}")
                rospy.signal_shutdown("Missing checkpoint file")
                return
        else:
            self.model_path = None
            
        self.bridge = CvBridge()
        
        # 3. 加载 RF-DETR 模型
        rospy.loginfo(f"Loading RF-DETR[Task: {self.task} | Size: {self.size}]...")
        self.model = self._load_model(self.task, self.size, self.model_path)
        
        if self.model is None:
            rospy.signal_shutdown("Failed to load model")
            return
            
        # (可选) 如果未来 Roboflow 官方支持 TensorRT/优化，可调用此方法
        if hasattr(self.model, "optimize_for_inference"):
            rospy.loginfo("Optimizing model for Jetson inference...")
            self.model.optimize_for_inference()
            
        rospy.loginfo("Model loaded successfully. Ready for inference!")
        
        # 4. 初始化发布者和订阅者
        self.pub = rospy.Publisher('/rf_detr/detections', DetectionArray, queue_size=1)
        self.sub = rospy.Subscriber(self.input_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        
    def _load_model(self, task, size, model_path):
        # 组装参数，如果有本地模型则使用 checkpoint 加载
        kwargs = {}
        if model_path:
            kwargs['checkpoint'] = model_path
            rospy.loginfo(f"Using local weights from: {model_path}")
        else:
            rospy.loginfo("Using default/downloaded weights.")
            
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
            rospy.logerr(f"Failed to instantiate model: {e}")
            
        rospy.logerr(f"Unsupported task ({task}) or size ({size}).")
        return None

    def image_callback(self, msg):
        try:
            # ROS Image -> OpenCV (RGB) -> PIL Image (因为模型需要 PIL 格式)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            pil_image = PILImage.fromarray(cv_image)
            
            # 执行推理 (返回 supervision.Detections 对象)
            detections_sv = self.model.predict(pil_image, threshold=self.conf_threshold)
            
            # 初始化输出的 ROS 消息
            out_msg = DetectionArray()
            out_msg.header = msg.header # 【细节】必须完美继承原图的时间戳，否则下游 visualizer 无法同步！
            
            # 解析 supervision 对象
            xyxy = detections_sv.xyxy
            confidences = detections_sv.confidence
            class_ids = detections_sv.class_id
            masks = detections_sv.mask
            
            # 只有当检测到物体时才进入循环
            if xyxy is not None and len(xyxy) > 0:
                for i in range(len(xyxy)):
                    det = Detection()
                    
                    # 赋值 BBox (直接转为 list 适配 float32[4])
                    det.bbox = xyxy[i].tolist()
                    
                    # 赋值 Score 与 Class ID (强制转为原生 float 和 int，防止 ROS 序列化报错)
                    det.score = float(confidences[i])
                    det.class_id = int(class_ids[i])
                    
                    # 匹配类别名称
                    if det.class_id < len(COCO_CLASSES):
                        det.class_name = COCO_CLASSES[det.class_id]
                    else:
                        det.class_name = str(det.class_id)
                    
                    # 赋值分割 Mask（如果是分割任务）
                    if masks is not None:
                        # 掩膜是 True/False，乘 255 转为 0/255 的图像
                        mask_uint8 = (masks[i] * 255).astype(np.uint8)
                        det.mask = self.bridge.cv2_to_imgmsg(mask_uint8, encoding="mono8")
                    
                    out_msg.detections.append(det)
                    
            # 发布检测结果 (哪怕没有检测到物体，也要发布一个空的数组，以保证下游画面持续刷新)
            self.pub.publish(out_msg)
            
        except Exception as e:
            # 【细节】使用 logerr_throttle 避免因画面持续报错导致终端被刷屏死机
            rospy.logerr_throttle(2.0, f"Error in inference callback: {e}")

if __name__ == '__main__':
    try:
        node = RFDetrNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
