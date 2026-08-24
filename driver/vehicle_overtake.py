#!/usr/bin/env python3
# yolo 모델을 이용해서 차를 감지하는 코드

import rospy
import cv2
import numpy as np
import torch
import time
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

class OneTimeOvertaker:
    def __init__(self):
        rospy.init_node('one_time_overtaker', anonymous=True)
        
        # 변수 먼저 초기화
        self.confidence_threshold = 0.3
        self.distance_threshold = 20.0
        self.focal_length = 800
        self.real_car_width = 1.8
        self.reference_line_ratio = 0.5
        
        # ROS 통신 설정
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw/", Image, self.image_callback, queue_size=1)
        self.current_lane_sub = rospy.Subscriber('/current_lane', String, self.lane_callback, queue_size=1)
        self.lane_change_pub = rospy.Publisher('/lane_change_command', String, queue_size=1)
        
        # YOLO 모델 설정
        self._setup_yolo_model()
        
        # 현재 차선 상태
        self.current_lane = "left"  # 초기값
        self.lane_detector_connected = False
        self.last_lane_update_time = rospy.Time.now()
        
        # 원샷 미션 상태
        self.detection_count = 0
        self.detection_threshold = 2  # 2번 탐지
        self.first_command_sent = False
        self.first_command_time = None
        self.second_command_sent = False
        self.mission_complete = False
        self.wait_duration = 9.0  # 8초 대기
        self.original_lane = None  # 추월 시작 전 원래 차선
        
        # 시각화
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        rospy.loginfo("=== One-Time Overtaker Started ===")
        rospy.loginfo("Mission: Overtake slow vehicle based on current lane")
        rospy.loginfo("Logic: Current lane → Opposite lane (8s) → Back to current lane")
        rospy.loginfo("Example: left → right → left  OR  right → left → right")

    def _setup_yolo_model(self):
        """YOLO 모델 설정"""
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
            self.model.eval()
            self.model.conf = self.confidence_threshold
            self.model.iou = 0.45
            self.model.classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
            rospy.loginfo("YOLOv5n model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load YOLO model: {e}")
            raise

    def image_callback(self, data):
        """메인 이미지 처리 콜백"""
        try:
            # 미션 완료시 더 이상 처리하지 않음
            if self.mission_complete:
                return
                
            # 이미지 변환
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            # 차량 탐지
            vehicles = self._detect_vehicles(cv_image)
            
            # 원샷 미션 로직 실행
            self._execute_one_shot_mission(vehicles, cv_image.shape)
            
            # 결과 시각화
            result_image = self._visualize_results(cv_image, vehicles)
            cv2.imshow('One-Time Overtaker', result_image)
            cv2.waitKey(1)
            
        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")

    def lane_callback(self, msg):
        """차선 상태 콜백"""
        # 미션 완료 후에는 아무것도 하지 않음
        if self.mission_complete:
            return
            
        lane_data = msg.data
        
        if lane_data in ["left", "right"]:
            if self.current_lane != lane_data:
                rospy.loginfo(f"Current lane updated: {self.current_lane} → {lane_data}")
                self.current_lane = lane_data
            
            self.lane_detector_connected = True
            self.last_lane_update_time = rospy.Time.now()

    def _detect_vehicles(self, image):
        """차량 탐지"""
        vehicles = []
        
        with torch.no_grad():
            results = self.model(image)
        
        detections = results.pandas().xyxy[0]
        
        for _, detection in detections.iterrows():
            if detection['confidence'] > self.confidence_threshold:
                x1, y1, x2, y2 = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']
                bbox_width = x2 - x1
                distance = (self.real_car_width * self.focal_length) / bbox_width if bbox_width > 0 else float('inf')
                
                vehicles.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': detection['confidence'],
                    'distance': distance,
                    'class_name': detection['name']
                })
        
        return vehicles

    def _is_vehicle_in_front(self, vehicle, image_shape):
        """차량이 내 앞에 있는지 확인 (중앙선 기준)"""
        image_width = image_shape[1]
        reference_line_x = int(image_width * self.reference_line_ratio)
        x1, y1, x2, y2 = vehicle['bbox']
        return x1 <= reference_line_x <= x2

    def _execute_one_shot_mission(self, vehicles, image_shape):
        """원샷 미션 로직 - 딱 2번만 publish"""
        
        # 미션 완료시 아무것도 하지 않음
        if self.mission_complete:
            return
        
        # 차선 정보가 없으면 대기
        if not self.lane_detector_connected:
            return
            
        # 첫 번째 명령 아직 안 보냄
        if not self.first_command_sent:
            # 앞에 있는 차량 확인
            front_vehicles = [v for v in vehicles 
                             if self._is_vehicle_in_front(v, image_shape) and 
                                v['distance'] <= self.distance_threshold]
            
            if front_vehicles:
                self.detection_count += 1
                rospy.loginfo(f"Vehicle detected! Count: {self.detection_count}/{self.detection_threshold}")
                
                if self.detection_count >= self.detection_threshold:
                    # 현재 차선 저장 및 추월 차선 결정
                    self.original_lane = self.current_lane
                    overtake_lane = "right" if self.current_lane == "left" else "left"
                    
                    # 첫 번째 명령 송신 (추월)
                    self.lane_change_pub.publish(String(data=overtake_lane))
                    self.first_command_sent = True
                    self.first_command_time = rospy.Time.now()
                    rospy.loginfo(f"FIRST COMMAND SENT: '{overtake_lane}' (overtake from {self.original_lane})")
            else:
                # 천천히 감소
                self.detection_count = max(0, self.detection_count - 1)
        
        # 첫 번째 명령 보냈지만 두 번째 명령 아직 안 보냄
        elif self.first_command_sent and not self.second_command_sent:
            # 8초 경과 확인
            elapsed_time = (rospy.Time.now() - self.first_command_time).to_sec()
            if elapsed_time >= self.wait_duration:
                # 두 번째 명령 송신 (원래 차선으로 복귀)
                self.lane_change_pub.publish(String(data=self.original_lane))
                self.second_command_sent = True
                rospy.loginfo(f"SECOND COMMAND SENT: '{self.original_lane}' (return to original)")
                
                # 미션 완료
                self.mission_complete = True
                rospy.loginfo("MISSION COMPLETE! No more commands will be sent.")

    def _visualize_results(self, image, vehicles):
        """결과 시각화"""
        result_image = image.copy()
        image_height, image_width = image.shape[:2]
        
        # 중앙선 그리기
        reference_line_x = int(image_width * self.reference_line_ratio)
        cv2.line(result_image, (reference_line_x, 0), (reference_line_x, image_height), (0, 0, 255), 3)
        cv2.putText(result_image, "CENTER", (reference_line_x - 30, 30), self.font, 0.6, (0, 0, 255), 2)
        
        # 상태 정보 표시
        self._draw_status_info(result_image)
        
        # 차량 정보 표시
        self._draw_vehicle_info(result_image, vehicles, image.shape)
        
        return result_image

    def _draw_status_info(self, image):
        """상태 정보 그리기"""
        y_offset = 60
        line_height = 30
        
        # 현재 차선 정보
        lane_color = (0, 255, 0) if self.lane_detector_connected else (0, 0, 255)
        cv2.putText(image, f"Current Lane: {self.current_lane}", (10, y_offset), self.font, 0.7, lane_color, 2)
        y_offset += line_height
        
        # 미션 상태
        if self.mission_complete:
            status_text = "MISSION COMPLETE"
            status_color = (0, 255, 0)
        elif self.second_command_sent:
            status_text = "RETURNING"
            status_color = (255, 255, 0)
        elif self.first_command_sent:
            elapsed = (rospy.Time.now() - self.first_command_time).to_sec()
            remaining = max(0, self.wait_duration - elapsed)
            status_text = f"WAITING ({remaining:.1f}s)"
            status_color = (255, 0, 255)
        else:
            status_text = "DETECTING"
            status_color = (0, 255, 255)
        
        cv2.putText(image, f"Status: {status_text}", (10, y_offset), self.font, 0.8, status_color, 2)
        y_offset += line_height
        
        # 탐지 카운터 (미션 완료 전에만)
        if not self.mission_complete:
            cv2.putText(image, f"Detection: {self.detection_count}/{self.detection_threshold}", 
                       (10, y_offset), self.font, 0.7, (255, 255, 255), 2)
            y_offset += line_height
        
        # 명령 상태
        command_status = []
        if self.first_command_sent and self.original_lane:
            overtake_lane = "right" if self.original_lane == "left" else "left"
            command_status.append(f"1st: '{overtake_lane}' ✓")
        if self.second_command_sent and self.original_lane:
            command_status.append(f"2nd: '{self.original_lane}' ✓")
        
        if command_status:
            cv2.putText(image, "Commands: " + " | ".join(command_status), 
                       (10, y_offset), self.font, 0.6, (100, 255, 100), 2)
            y_offset += line_height
        
        # 원래 차선 정보 (추월 시작 후)
        if self.original_lane and not self.mission_complete:
            cv2.putText(image, f"Original Lane: {self.original_lane}", 
                       (10, y_offset), self.font, 0.6, (255, 200, 100), 2)

    def _draw_vehicle_info(self, image, vehicles, image_shape):
        """차량 정보 그리기"""
        for i, vehicle in enumerate(vehicles):
            x1, y1, x2, y2 = vehicle['bbox']
            is_in_front = self._is_vehicle_in_front(vehicle, image_shape)
            
            # 색상 결정
            if is_in_front and vehicle['distance'] <= self.distance_threshold:
                color = (0, 0, 255)  # 빨간색 (타겟)
                status = "TARGET"
            elif is_in_front:
                color = (0, 165, 255)  # 주황색 (멀리)
                status = "FRONT"
            else:
                color = (0, 255, 0)  # 초록색 (옆)
                status = "SIDE"
            
            # 바운딩 박스
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            # 차량 정보
            info_lines = [
                f"Car #{i+1}",
                f"{status}",
                f"{vehicle['distance']:.1f}m"
            ]
            
            # 텍스트 배경
            text_bg_height = len(info_lines) * 15 + 5
            cv2.rectangle(image, (x1, y1-text_bg_height-5), (x1+100, y1), (0, 0, 0), -1)
            cv2.rectangle(image, (x1, y1-text_bg_height-5), (x1+100, y1), color, 2)
            
            # 텍스트 표시
            for j, line in enumerate(info_lines):
                cv2.putText(image, line, (x1+5, y1-text_bg_height+15*j+15), 
                           self.font, 0.4, (255, 255, 255), 1)

        # 미션 설명
        legend_y = image.shape[0] - 100
        cv2.putText(image, f"ONE-SHOT MISSION: 3 detections → overtake → 8s wait → return → DONE", 
                   (10, legend_y), self.font, 0.5, (255, 255, 255), 1)
        cv2.putText(image, "RED: Target Vehicle | Only 2 commands will ever be sent!", 
                   (10, legend_y + 20), self.font, 0.5, (255, 100, 100), 1)
        
        # 현재 차선에 따른 추월 방향 표시
        if self.current_lane and not self.mission_complete:
            overtake_direction = "right" if self.current_lane == "left" else "left"
            cv2.putText(image, f"Current: {self.current_lane} → Will overtake to: {overtake_direction}", 
                       (10, legend_y + 40), self.font, 0.5, (100, 255, 255), 1)

    def run(self):
        """메인 실행"""
        rospy.loginfo("One-Time Overtaker running...")
        rospy.loginfo("Mission: Send exactly 2 commands based on current lane and stop!")
        rospy.loginfo("Will overtake to opposite lane, then return to original")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down...")
        finally:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        overtaker = OneTimeOvertaker()
        overtaker.run()
    except rospy.ROSInterruptException:
        pass