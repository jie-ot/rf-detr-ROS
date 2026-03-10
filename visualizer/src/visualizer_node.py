#!/usr/bin/env python3
import rospy
import cv2
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rf_detr.msg import DetectionArray

class VisualizerNode:
    def __init__(self):
        self.bridge = CvBridge()
        sub_img = message_filters.Subscriber('/camera/image_raw', Image, queue_size=10)
        sub_det = message_filters.Subscriber('/detections', DetectionArray, queue_size=10)
        
        # 时间同步器
        ts = message_filters.ApproximateTimeSynchronizer([sub_img, sub_det], queue_size=10, slop=0.05)
        ts.registerCallback(self.callback)
        
        self.pub = rospy.Publisher('/visualizer', Image, queue_size=10)
        rospy.loginfo("[Visualizer] Node started.")

    def callback(self, img_msg, det_msg):
        img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        h_img, w_img = img.shape[:2]
        
        # 画框
        for det in det_msg.detections:
            x1 = int(max(0, det.bbox[0]))
            y1 = int(max(0, det.bbox[1]))
            x2 = int(min(w_img, det.bbox[2]))
            y2 = int(min(h_img, det.bbox[3]))
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{det.class_name} {det.score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 发布
        out_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        out_msg.header = img_msg.header
        self.pub.publish(out_msg)

if __name__ == '__main__':
    rospy.init_node('visualizer')
    VisualizerNode()
    rospy.spin()
