#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2, rospy, numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge

bridge = CvBridge()
cv_image = np.empty(shape=[0])

def img_callback(data):
    global cv_image
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

def detect_traffic_light_color(img):
    # 신호등은 위에 있으니까 오류 방지를 위해 반으로 자르고 위에 부분만 사용
    h, _, _ = img.shape
    roi = img[0:h//2,:]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 빨간색 범위
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # 노란색 범위
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # 초록색 범위
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([90, 255, 255])

    # 마스크 생성
    red_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    red_pixels = cv2.countNonZero(red_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    green_pixels = cv2.countNonZero(green_mask)

    threshold = 500

    # 초록색이면 출발
    if green_pixels > threshold:
        return "Go", roi
    # 노란색과 빨간색일때는 정지해야 함
    elif (yellow_pixels > threshold) | (red_pixels > threshold):
        return "Stop", roi
    else:
        return "None", roi

# ROS 노드 초기화
rospy.init_node('cam_test', anonymous=True)

# 이미지 수신하기 위해 토픽 Subscribe
rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback)

# "traffic_is_go", "traffic_is_stop" 토픽 publish
go_pub = rospy.Publisher("traffic_is_go", Int32, queue_size=1)
stop_pub = rospy.Publisher("traffic_is_stop", Int32, queue_size=1)

# 이미지 데이터 오기 전까지 대기기
rospy.wait_for_message("/usb_cam/image_raw/", Image)
print("Camera Ready --------------")

while not rospy.is_shutdown():
    if cv_image.size != 0:
        color, roi = detect_traffic_light_color(cv_image)

        # 초록색이면 traffic_is_go를를 1, 아니면 0
        if color == "Go":
            go_pub.publish(1)
        else:
            go_pub.publish(0)

        # 빨간색이거나 노란색이면 traffic_is_stop을을 1, 아니면 0
        if color == "Stop":
            stop_pub.publish(1)
        else:
            stop_pub.publish(0)

        #cv2.imshow("original", roi)
        #cv2.waitKey(1)
