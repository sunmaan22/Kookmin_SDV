# -*- coding: utf-8 -*-
import numpy as np
import cv2
import rospy
import time
import os
from collections import deque
from sensor_msgs.msg import Image, LaserScan
from xycar_msgs.msg import XycarMotor
from std_msgs.msg import Int32, Bool, String
from cv_bridge import CvBridge
import matplotlib.pyplot as plt

class LaneDetector:
    def __init__(self):
        # ROS 관련 변수
        self.image = np.empty(shape=[0])
        self.ranges = None  # 라이다 데이터 저장용 (나중에 구현)
        self.motor = None
        self.motor_msg = XycarMotor()
        self.bridge = CvBridge()
        
        # 적응형 PID 제어 변수
        self.prev_error = 0
        self.integral = 0
        
        # PID 게인 기준값
        self.kp_base = 0.35
        self.ki_base = 0.001
        self.kd_base = 0.3
        
        # 상황에 따라 바뀔 PID 게인 
        self.kp = self.kp_base
        self.ki = self.ki_base
        self.kd = self.kd_base
        
        # 차선 추적 관련 변수
        self.lane_history = deque(maxlen=5)  # 이전 차선 위치 기록
        self.lane_detected = False
        self.last_lane_position = None
        self.confidence = 0
        self.is_cone_mode = False # 추가
        
        # 차선 변경 관련 변수
        self.current_lane = "left"
        self.yellow_line_position = None
        self.left_white_position = None
        self.right_white_position = None
        self.yellow_detected = False
        self.left_white_detected = False
        self.right_white_detected = False
        
        # 차선 기준 오프셋 설정
        self.lane_offset_left = -200  # 왼쪽 차선과의 거리 증가
        self.lane_offset_right = 200  # 오른쪽 차선과의 거리 증가
        
        # 차선 변경 제어 변수
        self.is_lane_changing = False  # 차선 변경 중인지 상태
        self.lane_change_command = None  # "left" 또는 "right"
        self.lane_change_start_time = 0  # 차선 변경 시작 시간
        self.lane_change_phase = 0  # 0: 대기, 1: 주 회전, 2: 반대 회전, 3: 직진 안정화
        
        # 왼쪽 차선 변경 설정
        self.lane_change_turn_duration_left = 0.8  # 왼쪽 주 회전 시간 (초)
        self.lane_change_counter_duration_left = 0.5  # 왼쪽 반대 회전 시간 (초)
        self.lane_change_stabilize_duration_left = 0.2  # 왼쪽 직진 안정화 시간 (초)
        self.lane_change_angle_left = 45  # 왼쪽 차선 변경 시 조향각
        self.lane_change_counter_angle_left = 20  # 왼쪽 반대 회전 조향각
        self.lane_change_speed_left = 45.0  # 왼쪽 차선 변경 시 속도
       
        # 오른쪽 차선 변경 설정
        self.lane_change_turn_duration_right = 0.9  # 오른쪽 주 회전 시간 (초)
        self.lane_change_counter_duration_right = 0.55  # 오른쪽 반대 회전 시간 (초)
        self.lane_change_stabilize_duration_right = 0.2  # 오른쪽 직진 안정화 시간 (초)
        self.lane_change_angle_right = 45  # 오른쪽 차선 변경 시 조향각
        self.lane_change_counter_angle_right = 20  # 오른쪽 반대 회전 조향각
        self.lane_change_speed_right = 45.0  # 오른쪽 차선 변경 시 속도
        
        # 라이다 관련 변수 (나중에 구현)
        self.obstacle_detected = False
        self.obstacle_direction = 0
        self.obstacle_distance = 0
        
        # 속도 제어 변수 
        self.base_speed = 50.0  
        self.max_speed = 55.0   
        self.min_speed = 45.0   
        
        # 곡률 계산 변수
        self.current_curvature = 0
        self.curvature_history = deque(maxlen=3)
        
        # 신호등 인식 관련 변수
        self.traffic_light_color = "None"
        self.is_red = False
        self.is_green = False
        self.red_frames = 0
        self.green_frames = 0
        self.light_threshold = 2  # 신호등 감지 프레임 수 감소 (3 -> 2)
        
        # 디버그용 시각화 설정
        self.debug_mode = True
        
        # 성능 개선을 위한 변수
        self.skip_frames = 0  # 프레임 처리 빈도 조절용
        self.process_every_n_frames = 1  # 매 n 프레임마다 처리 (1로 설정하면 모든 프레임 처리)
        
        # ROI 설정 최적화
        self.roi_height_percent = 0.35  # ROI 높이 비율
    
    def usbcam_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    
    def lidar_callback(self, data):
        # 라이다 데이터 처리 (나중에 구현)
        pass
    
    def drive(self, angle, speed):
        self.motor_msg.angle = float(angle)
        self.motor_msg.speed = float(speed)
        self.motor.publish(self.motor_msg)
    
    # 차선 변경 콜백 함수
    def lane_change_callback(self, msg): 
        command = msg.data.lower()
        
        # 이미 차선 변경 중이면 무시
        if self.is_lane_changing:
            rospy.logwarn(f"Already changing lanes, ignoring command: {command}")
            return
        
        # 유효한 명령인지 확인
        if command in ["left", "right"]:
            rospy.loginfo(f"Lane change command received: {command}")
            self.start_lane_change(command)
        else:
            rospy.logwarn(f"Invalid lane change command: {command}")
    
    def get_lane_change_params(self, direction):
        # 방향에 따른 차선 변경 파라미터 반환
        if direction == "left":
            return {
                'turn_duration': self.lane_change_turn_duration_left,
                'counter_duration': self.lane_change_counter_duration_left,
                'stabilize_duration': self.lane_change_stabilize_duration_left,
                'angle': self.lane_change_angle_left,
                'counter_angle': self.lane_change_counter_angle_left,
                'speed': self.lane_change_speed_left
            }
        else:  # "right"
            return {
                'turn_duration': self.lane_change_turn_duration_right,
                'counter_duration': self.lane_change_counter_duration_right,
                'stabilize_duration': self.lane_change_stabilize_duration_right,
                'angle': self.lane_change_angle_right,
                'counter_angle': self.lane_change_counter_angle_right,
                'speed': self.lane_change_speed_right
            }
    
    def start_lane_change(self, direction):
        """차선 변경 시작"""
        self.is_lane_changing = True
        self.lane_change_command = direction
        self.lane_change_start_time = time.time()
        self.lane_change_phase = 1  # 주 회전 단계부터 시작
        
        # 목표 차선 설정
        if direction == "right":
            self.current_lane = "right"
        else:
            self.current_lane = "left"
        
        # 해당 방향의 파라미터 가져오기
        params = self.get_lane_change_params(direction)
        
        rospy.loginfo(f"Starting 3-phase lane change to {direction}")
        rospy.loginfo(f"Phase 1: Main turn ({params['turn_duration']}s, angle: {params['angle']}°)")
        rospy.loginfo(f"Phase 2: Counter steer ({params['counter_duration']}s, angle: {params['counter_angle']}°)")
        rospy.loginfo(f"Phase 3: Stabilize ({params['stabilize_duration']}s)")
        rospy.loginfo(f"Speed: {params['speed']} km/h")
    
    def update_lane_change(self):
        # 차선 변경 상태 업데이트 - 3단계 프로세스 (좌우 개별 파라미터 적용)
        if not self.is_lane_changing:
            return False  # 차선 변경 중이 아니면 False 반환
        
        current_time = time.time()
        elapsed_time = current_time - self.lane_change_start_time
        
        # 현재 방향에 맞는 파라미터 가져오기
        params = self.get_lane_change_params(self.lane_change_command)
        
        if self.lane_change_phase == 1:  # 주 회전 단계
            if elapsed_time < params['turn_duration']:
                # 지정된 방향으로 핸들 꺾기
                if self.lane_change_command == "right":
                    angle = params['angle']  # 오른쪽은 양수
                else:
                    angle = -params['angle']   # 왼쪽은 음수
                
                speed = params['speed']
                
                # 신호등이 빨간색이면 정지
                if self.traffic_light_color == "Red":
                    speed = 0.0
                
                self.drive(angle, speed)
                rospy.loginfo_throttle(0.5, f"Lane changing - Phase 1: Main turn {self.lane_change_command} ({angle}°)")
                return True
            else:
                # 반대 회전 단계로 전환
                self.lane_change_phase = 2
                self.lane_change_start_time = current_time
                rospy.loginfo("Lane change - Phase 2: Counter steering")
        
        elif self.lane_change_phase == 2:  # 반대 회전 단계 (안정화)
            if elapsed_time < params['counter_duration']:
                # 반대 방향으로 살짝 회전 (안정화)
                if self.lane_change_command == "right":
                    angle = -params['counter_angle']  # 오른쪽 변경 후 왼쪽으로 살짝
                else:
                    angle = params['counter_angle']  # 왼쪽 변경 후 오른쪽으로 살짝
                
                speed = params['speed']
                
                # 신호등이 빨간색이면 정지
                if self.traffic_light_color == "Red":
                    speed = 0.0
                
                self.drive(angle, speed)
                rospy.loginfo_throttle(0.5, f"Lane changing - Phase 2: Counter steering ({angle}°)")
                return True
            else:
                # 직진 안정화 단계로 전환
                self.lane_change_phase = 3
                self.lane_change_start_time = current_time
                rospy.loginfo("Lane change - Phase 3: Final stabilizing")
        
        elif self.lane_change_phase == 3:  # 직진 안정화 단계
            if elapsed_time < params['stabilize_duration']:
                # 직진으로 안정화
                angle = 0
                speed = params['speed']
                
                # 신호등이 빨간색이면 정지
                if self.traffic_light_color == "Red":
                    speed = 0.0
                
                self.drive(angle, speed)
                rospy.loginfo_throttle(0.5, "Lane changing - Phase 3: Final stabilizing")
                return True
            else:
                # 차선 변경 완료
                self.complete_lane_change()
                return False
        
        return True
    
    def complete_lane_change(self):
        # 차선 변경 완료
        self.is_lane_changing = False
        self.lane_change_command = None
        self.lane_change_phase = 0
        
        # PID 제어 변수 초기화 (안정적인 복귀를 위해)
        self.prev_error = 0
        self.integral = 0
        
        rospy.loginfo(f"3-phase lane change completed! Now in {self.current_lane} lane")
        rospy.loginfo("Returning to normal lane following mode")
    
    def publish_current_lane(self):
        # 현재 차선 상태 퍼블리시
        lane_msg = String()
        lane_msg.data = self.current_lane
        self.current_lane_pub.publish(lane_msg)
    
    def adaptive_pid_control(self, error):
        # 속도와 곡률에 따라 PID 게인을 동적으로 조정하는 적응형 PID 제어
        # 현재 곡률 기반으로 PID 게인 조정
        curvature_factor = min(1.5, 1.0 + abs(self.current_curvature) * 50)
        
        # 높은 곡률에서는 P게인을 높이고 D게인을 낮춤 (더 민감하게 반응)
        self.kp = self.kp_base * curvature_factor
        self.kd = self.kd_base / curvature_factor
        
        # 신뢰도가 낮을 때는 I항을 줄임
        confidence_factor = max(0.1, self.confidence)
        self.ki = self.ki_base * confidence_factor
        
        # PID 제어 계산
        derivative = error - self.prev_error
        self.integral += error
        
        # Anti-windup: integral 값 제한
        self.integral = np.clip(self.integral, -20, 20)  # 적분 항 범위 축소 (더 빠른 응답)
        
        self.prev_error = error
        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        return control
    
    def get_lane_positions(self, image):
        height, width = image.shape[:2]
        
        # ROI 설정 (하단 35% - 처리량 감소)
        roi_height = int(height * self.roi_height_percent)
        roi = image[height - roi_height:height, :]
        
        # 이미지 전처리 - 가우시안 블러 커널 크기 감소로 속도 향상
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)  # 5x5 -> 3x3으로 감소
        
        # 색상 분리 (HSV 공간)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 흰색 차선 마스크 - 민감도 증가 및 최적화
        white_lower = np.array([0, 0, 170])
        white_upper = np.array([180, 40, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        # 노란색 차선 마스크 - 최적화
        yellow_lower = np.array([18, 80, 80])
        yellow_upper = np.array([32, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # 에지 검출 - 계산량 감소를 위해 임계값 조정
        edges = cv2.Canny(blur, 70, 170)  # 임계값 증가로 노이즈 감소
        
        # 마스크와 에지 결합
        white_edges = cv2.bitwise_and(edges, edges, mask=white_mask)
        yellow_edges = cv2.bitwise_and(edges, edges, mask=yellow_mask)
        
        # 허프 변환으로 선 검출 - 파라미터 최적화
        white_lines = self.detect_lines(white_edges)
        yellow_lines = self.detect_lines(yellow_edges)
        
        # 차선 위치 계산 (왼쪽 흰색, 오른쪽 흰색, 노란색)
        left_white, right_white = self.find_white_lane_positions(white_lines, width)
        yellow_position = self.find_yellow_lane_position(yellow_lines, width)
        
        # 차선 감지 상태 업데이트
        self.left_white_detected = left_white is not None
        self.right_white_detected = right_white is not None
        self.yellow_detected = yellow_position is not None
        
        # 검출된 차선 위치 저장
        self.left_white_position = left_white
        self.right_white_position = right_white
        self.yellow_line_position = yellow_position
        
        # 차선 변경 중이 아닐 때만 자동 차선 변경 로직 수행
        if not self.is_lane_changing:
            self.update_lane_change_state(width)
        
        # 사용할 차선 위치 결정
        if self.current_lane == "left":
            lane_position = self.left_white_position
            self.lane_detected = self.left_white_detected
        else:  # "right"
            lane_position = self.right_white_position
            self.lane_detected = self.right_white_detected
        
        # 차선이 감지되지 않으면 이전 상태 활용
        if lane_position is None:
            if len(self.lane_history) > 0:
                lane_position = self.lane_history[-1]
                self.confidence -= 0.1  # 신뢰도 더 빠르게 감소
            else:
                # 처음 시작 시 기본값 설정
                if self.current_lane == "left":
                    lane_position = width // 2 + self.lane_offset_left
                else:
                    lane_position = width // 2 + self.lane_offset_right
                self.confidence = 0
            self.lane_detected = False
        else:
            self.confidence = min(1.0, self.confidence + 0.1)  # 신뢰도 더 빠르게 증가
            
            # 갑작스러운 변화 필터링 (노이즈 제거) - 더 빠른 응답을 위해 조정
            if self.last_lane_position is not None:
                # 신뢰도가 낮을수록 더 많은 변화 허용
                max_change = 30 * (1.0 - self.confidence*0.5)  # 더 큰 변화 허용
                if abs(lane_position - self.last_lane_position) > max_change:
                    # 변화가 너무 크면 이전 위치에 가중치 부여 (전보다 적은 가중치)
                    lane_position = int(0.7 * self.last_lane_position + 0.3 * lane_position)
            
            self.last_lane_position = lane_position
            self.lane_history.append(lane_position)
        
        # 곡률 계산
        curve = 0
        if white_lines is not None and len(white_lines) > 0:
            curve = self.calculate_curvature(white_lines)
            self.curvature_history.append(curve)
            self.current_curvature = np.mean(self.curvature_history)
        
        # 디버그용 마스크 결합
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        return lane_position, self.lane_detected, white_lines, yellow_lines, combined_mask
    
    def detect_lines(self, edges):
        # 허프 변환 파라미터 최적화 - 더 적은 선을 감지하도록 임계값 증가
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25, minLineLength=15, maxLineGap=12)
        return lines
    
    def find_white_lane_positions(self, white_lines, width):
        # 흰색 차선 감지 - 왼쪽과 오른쪽 모두 감지
        if white_lines is None:
            return None, None
        
        # 왼쪽/오른쪽 차선 후보
        left_candidates = []
        right_candidates = []
        
        # 이미지 중앙 지점
        mid_x = width // 2
        
        for line in white_lines:
            x1, y1, x2, y2 = line[0]
            # 선의 중점 계산
            mid_point_x = (x1 + x2) // 2
            
            # 이미지 왼쪽에 있는 선은 왼쪽 차선 후보
            if mid_point_x < mid_x - 40:
                left_candidates.append(line[0])
            # 이미지 오른쪽에 있는 선은 오른쪽 차선 후보
            elif mid_point_x > mid_x + 40:
                right_candidates.append(line[0])
        
        # 왼쪽 차선 위치 계산
        left_white = None
        if left_candidates:
            left_x = []
            for line in left_candidates:
                x1, y1, x2, y2 = line
                left_x.extend([x1, x2])
            left_white = int(np.mean(left_x))
            
            # 추가: 특정 값 이하의 이상치 필터링 (노이즈 제거)
            if left_white < 30:  # 화면 너무 왼쪽 가장자리의 값은 무시
                left_white = None
        
        # 오른쪽 차선 위치 계산
        right_white = None
        if right_candidates:
            right_x = []
            for line in right_candidates:
                x1, y1, x2, y2 = line
                right_x.extend([x1, x2])
            right_white = int(np.mean(right_x))
            
            # 추가: 특정 값 이상의 이상치 필터링 (노이즈 제거)
            if right_white > width - 30:  # 화면 너무 오른쪽 가장자리의 값은 무시
                right_white = None
        
        return left_white, right_white
    
    def find_yellow_lane_position(self, yellow_lines, width):
        # 노란색 중앙선 위치 계산
        if yellow_lines is None:
            return None
        
        # 노란색 차선 후보
        yellow_candidates = []
        
        for line in yellow_lines:
            yellow_candidates.append(line[0])
        
        if not yellow_candidates:
            return None
        
        # 검출된 노란선들의 x좌표 평균을 차선 위치로 사용
        yellow_x = []
        for line in yellow_candidates:
            x1, y1, x2, y2 = line
            yellow_x.extend([x1, x2])
        
        if not yellow_x:
            return None
        
        yellow_position = int(np.mean(yellow_x))
        return yellow_position
    
    def update_lane_change_state(self, width):
        # 차량의 현재 차선 상태를 업데이트
        # 이미지 중앙점 (차량의 중심선)
        car_center_x = width // 2
        
        # 노란선이 감지된 경우
        if self.yellow_detected:
            # 차량 중심이 노란선을 완전히 넘어섰는지 확인
            # 노란선이 차량 중심부에서 왼쪽에 있으면 오른쪽 차선으로 변경
            if self.yellow_line_position < car_center_x - 20:
                self.current_lane = "right"
            # 노란선이 차량 중심부에서 오른쪽에 있으면 왼쪽 차선으로 변경
            elif self.yellow_line_position > car_center_x + 20:
                self.current_lane = "left"
        
        # 노란선이 감지되지 않았지만 양쪽 흰색 차선이 모두 감지된 경우
        elif self.left_white_detected and self.right_white_detected:
            # 양쪽 차선간 거리 확인 (너무 좁으면 단일 차선으로 판단)
            lane_width = abs(self.right_white_position - self.left_white_position)
            if lane_width > 180:  # 충분히 넓은 경우에만 차선 변경 고려
                # 왼쪽 차선과의 거리와 오른쪽 차선과의 거리를 비교하여 더 가까운 쪽으로 설정
                dist_to_left = abs(car_center_x - self.left_white_position)
                dist_to_right = abs(car_center_x - self.right_white_position)
                
                if dist_to_left < dist_to_right:
                    self.current_lane = "left"
                else:
                    self.current_lane = "right"
    
    def calculate_curvature(self, lines):
        if lines is None or len(lines) < 2:
            return 0
        
        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.append((x1, y1))
            points.append((x2, y2))
        
        # 최소 3개의 점이 필요함
        if len(points) < 3:
            return 0
        
        # 포인트를 y 좌표로 정렬
        points.sort(key=lambda p: p[1])
        
        # 포물선 피팅 (2차 다항식)
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        
        try:
            coeffs = np.polyfit(y, x, 2)
            # 첫 번째 계수가 곡률과 관련됨
            return coeffs[0]
        except:
            return 0
    
    def detect_traffic_light(self, image):
        # 신호등 색상 감지 함수 - 성능 최적화
        # 매 프레임마다 처리하지 않고 특정 주기로만 처리
        self.skip_frames += 1
        if self.skip_frames % 3 != 0:  # 매 3프레임마다 처리
            # 이전 상태 반환
            if self.is_green:
                return "Green"
            elif self.is_red:
                return "Red"
            else:
                return "None"
        
        if image.size == 0:
            return "None"
        
        # 상단 ROI 설정 (신호등은 보통 상단에 위치)
        height, width = image.shape[:2]
        roi_height = int(height * 0.3)  # 상단 30%만 처리 (40% -> 30%)
        roi = image[0:roi_height, :]
        
        # 이미지 크기 축소로 처리 속도 향상
        roi = cv2.resize(roi, (width//2, roi_height//2))
        
        # HSV 색상 공간으로 변환
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 빨간색 범위 (색상환 양 끝에 위치하므로 두 영역으로 분리)
        lower_red1 = np.array([0, 140, 140])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 140, 140])
        upper_red2 = np.array([180, 255, 255])
        
        # 초록색 범위
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([90, 255, 255])
        
        # 마스크 생성 (노란색은 제외하여 계산량 감소)
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # 각 색상별 픽셀 수 계산
        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)
        
        # 신호등 색상 판단
        threshold = 300  # 픽셀 수 임계값
        
        # 일시적인 색상 감지로 노이즈 필터링
        if green_pixels > threshold:
            self.green_frames += 1
            self.red_frames = 0
        elif red_pixels > threshold:  # 노란색은 제외
            self.red_frames += 1
            self.green_frames = 0
        else:
            # 어떤 색상도 감지되지 않으면 카운터 감소
            self.green_frames = max(0, self.green_frames - 1)
            self.red_frames = max(0, self.red_frames - 1)
        
        # 연속 프레임 기준으로 신호등 상태 결정
        if self.green_frames >= self.light_threshold:
            self.is_green = True
            self.is_red = False
            return "Green"
        elif self.red_frames >= self.light_threshold:
            self.is_red = True
            self.is_green = False
            return "Red"
        else:
            # 충분한 프레임이 감지되지 않으면 이전 상태 유지
            if self.is_green:
                return "Green"
            elif self.is_red:
                return "Red"
            else:
                return "None"
    
    def visualize(self, image, lane_position, white_lines, yellow_lines, combined_mask):
        if not self.debug_mode:
            return
        
        # 디버그 이미지 생성
        debug_img = image.copy()
        height, width = debug_img.shape[:2]
        roi_height = int(height * self.roi_height_percent)
        
        # ROI 표시
        cv2.rectangle(debug_img, (0, height - roi_height), (width, height), (0, 255, 0), 2)
        
        # 차선 위치 표시
        cv2.circle(debug_img, (lane_position, height - roi_height//2), 10, (0, 255, 255), -1)
        
        # 현재 기준 차선 표시
        if self.current_lane == "left":
            # 왼쪽 흰색 차선 기준 주행
            desired_position = width // 2 + self.lane_offset_left
            lane_color = (255, 255, 0)  # 청록색
        else:
            # 오른쪽 흰색 차선 기준 주행
            desired_position = width // 2 + self.lane_offset_right
            lane_color = (0, 165, 255)  # 주황색
            
        cv2.circle(debug_img, (desired_position, height - roi_height//2), 5, (255, 0, 0), -1)
        cv2.line(debug_img, (width // 2, height - roi_height//2), 
                (desired_position, height - roi_height//2), (255, 0, 0), 2)
        
        # 감지된 차선 표시 (왼쪽 흰색, 오른쪽 흰색, 노란색)
        if self.left_white_position is not None:
            cv2.circle(debug_img, (self.left_white_position, height - roi_height//2), 7, (255, 255, 0), -1)
        if self.right_white_position is not None:
            cv2.circle(debug_img, (self.right_white_position, height - roi_height//2), 7, (0, 165, 255), -1)
        if self.yellow_line_position is not None:
            cv2.circle(debug_img, (self.yellow_line_position, height - roi_height//2), 7, (0, 255, 255), -1)
        
        # 선 그리기
        if white_lines is not None:
            for line in white_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(debug_img, (x1, y1 + height - roi_height), (x2, y2 + height - roi_height), (255, 255, 255), 2)
        if yellow_lines is not None:
            for line in yellow_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(debug_img, (x1, y1 + height - roi_height), (x2, y2 + height - roi_height), (0, 255, 255), 2)
        
        # 디버그 정보 표시
        cv2.putText(debug_img, f"Lane Position: {lane_position}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Current Lane: {self.current_lane}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, lane_color, 2)
        cv2.putText(debug_img, f"Confidence: {self.confidence:.2f}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Curvature: {self.current_curvature:.5f}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Speed: {self.motor_msg.speed:.1f}", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 차선 변경 상태 표시 (개별 파라미터 정보 포함)
        if self.is_lane_changing:
            phase_names = {1: "MAIN TURN", 2: "COUNTER STEER", 3: "STABILIZE"}
            phase_name = phase_names.get(self.lane_change_phase, "UNKNOWN")
            status_text = f"LANE CHANGING to {self.lane_change_command.upper()} - {phase_name}"
            cv2.putText(debug_img, status_text, (10, 180), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 현재 사용 중인 파라미터 표시
            params = self.get_lane_change_params(self.lane_change_command)
            param_text = f"Params: A={params['angle']}°, S={params['speed']}"
            cv2.putText(debug_img, param_text, (10, 210), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 신호등 상태 표시
        light_color = (0, 0, 0)  # 기본 검정색
        if self.traffic_light_color == "Red":
            light_color = (0, 0, 255)  # 빨간색
            cv2.putText(debug_img, "RED LIGHT", (width//2 - 100, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, light_color, 2)
        
        # 이미지 표시 (Main 화면만)
        cv2.imshow("Main", debug_img)
        cv2.waitKey(1)
    
    def calculate_adaptive_speed(self, angle, width):
        # 곡률, 차선 신뢰도, 조향각 등에 기반한 비선형 속도 계산
        # 기본 속도 설정
        speed = self.base_speed
        
        # 1. 곡률 기반 속도 조정 (비선형) - 곡선에서도 더 빠르게 주행
        curvature_magnitude = abs(self.current_curvature)
        if curvature_magnitude > 0.001:  # 곡률이 있는 경우
            # 비선형 관계: 곡률이 커질수록 속도가 점진적으로 감소
            curve_factor = np.exp(-80 * curvature_magnitude)  # 지수 함수 조정 (100 -> 80)
            speed *= max(0.7, curve_factor)  # 최소 70%까지만 감속 (더 빠르게)
        
        # 2. 조향각 기반 속도 조정 (비선형) - 더 급한 커브에서도 속도 유지
        abs_angle = abs(angle)
        if abs_angle > 15:  # 어느 정도 꺾는 경우에만 감속 (10 -> 15로 증가)
            # 비선형 관계: 각도가 커질수록 속도가 점진적으로 감소
            angle_factor = 1.0 - (abs_angle / 70.0)**1.2  # 지수 1.5 -> 1.2로 감소, 분모 증가
            speed *= max(0.6, angle_factor)  # 최소 60%까지 감속 (전보다 덜 감속)
        
        # 3. 차선 신뢰도 기반 속도 조정 - 신뢰도가 낮아도 더 빠르게
        if self.confidence < 0.6:  # 신뢰도가 매우 낮을 때만 감속 (0.7 -> 0.6)
            confidence_factor = 0.8 + 0.2 * self.confidence  # 신뢰도에 따른 가중치 (80%~100%)
            speed *= confidence_factor
        
        # 4. 차선 변경 중 속도 조정 - 차선 변경 중에도 속도 유지
        if self.yellow_detected:
            # 차선 변경 중에는 약간만 속도 감소
            speed *= 0.9  # 80% -> 90%로 증가
        
        # 5. 신호등 감지 시 속도 조정
        if self.traffic_light_color == "Red":
            # 빨간 신호등 감지 시 점진적 감속
            speed = 0  # 완전히 정지
        
        # 6. 장애물 감지 시 속도 조정 (나중에 구현)
        if self.obstacle_detected:
            # 장애물까지의 거리에 따라 속도 조정
            distance_factor = min(1.0, self.obstacle_distance / 4.0)  # 5m -> 4m로 감소
            speed *= max(0.5, distance_factor)  # 최소 50%까지 감속 (40% -> 50%로 증가)
        
        # 최종 속도 제한
        speed = np.clip(speed, self.min_speed, self.max_speed)
        
        return speed
    
    # --- 새로 추가 ---
    def cone_mode_callback(self, msg: Bool):
        # Pure-Pursuit 모드(on = True) 여부 저장
        self.is_cone_mode = msg.data
    
    def calculate_driving_commands(self, lane_position, width):
        # 차량 중심에서 차선까지의 이상적인 거리 (오프셋)
        if self.current_lane == "left":
            desired_position = width // 2 + self.lane_offset_left
        else:  # "right"
            desired_position = width // 2 + self.lane_offset_right
        
        # 에러 계산 (현재 차선 위치와 목표 위치의 차이)
        error = lane_position - desired_position
        
        # 적응형 PID 제어로 조향각 계산
        angle = self.adaptive_pid_control(error)
        
        # 조향각 제한
        angle = np.clip(angle, -50, 50)  # -50~50 도
        
        # 비선형 속도 계산
        speed = self.calculate_adaptive_speed(angle, width)
        
        # 신호등이 빨간색이면 정지
        if self.traffic_light_color == "Red":
            speed = 0.0
        
        return angle, speed
    
    def run(self):
        # ROS 초기화
        rospy.init_node('Lane_Detector')
        rospy.Subscriber("/usb_cam/image_raw/", Image, self.usbcam_callback, queue_size=1)
        rospy.Subscriber("/scan", LaserScan, self.lidar_callback, queue_size=1)
        self.motor = rospy.Publisher('xycar_motor', XycarMotor, queue_size=1)
        rospy.Subscriber("/is_cone_mode", Bool, self.cone_mode_callback, queue_size=1) # 추가
        
        # === 차선 변경 관련 ROS 설정 ===
        rospy.Subscriber("/lane_change_command", String, self.lane_change_callback, queue_size=1)
        self.current_lane_pub = rospy.Publisher('/current_lane', String, queue_size=1)
        
        # 첫 메시지 대기s
        rospy.loginfo("Waiting for camera and lidar data...")
        rospy.wait_for_message("/usb_cam/image_raw/", Image)
        # 라이다 데이터는 나중에 구현하므로 대기하지 않음
        rospy.loginfo("Received first data, starting lane detection...")
        
        # === 차선 변경 파라미터 로그 출력 ===
        rospy.loginfo("=== Left Lane Change Parameters ===")
        rospy.loginfo(f"Turn Duration: {self.lane_change_turn_duration_left}s")
        rospy.loginfo(f"Counter Duration: {self.lane_change_counter_duration_left}s")
        rospy.loginfo(f"Stabilize Duration: {self.lane_change_stabilize_duration_left}s")
        rospy.loginfo(f"Turn Angle: {self.lane_change_angle_left}°")
        rospy.loginfo(f"Counter Angle: {self.lane_change_counter_angle_left}°")
        rospy.loginfo(f"Speed: {self.lane_change_speed_left}")
        
        rospy.loginfo("=== Right Lane Change Parameters ===")
        rospy.loginfo(f"Turn Duration: {self.lane_change_turn_duration_right}s")
        rospy.loginfo(f"Counter Duration: {self.lane_change_counter_duration_right}s")
        rospy.loginfo(f"Stabilize Duration: {self.lane_change_stabilize_duration_right}s")
        rospy.loginfo(f"Turn Angle: {self.lane_change_angle_right}°")
        rospy.loginfo(f"Counter Angle: {self.lane_change_counter_angle_right}°")
        rospy.loginfo(f"Speed: {self.lane_change_speed_right}")
        
        rate = rospy.Rate(30)  # 30Hz
        
        # 프레임 스킵 변수
        frame_counter = 0
        process_every_n = 1  # 매 n 프레임마다 처리
        
        # 신호등 인식과 주행 시작을 위한 초기화 단계
        initialization_phase = True
        initialization_frames = 0
        required_init_frames = 5  # 최소 5프레임의 인식 절차를 수행
        
        # 처음에는 반드시 정지 상태로 시작
        self.drive(0, 0)
        rospy.loginfo("Starting with stopped state for safety...")
        
        while not rospy.is_shutdown():
            if self.image.size == 0:
                rate.sleep()
                continue

            if self.is_cone_mode: #추가
                rate.sleep()
                continue
            
            # 이미지 복사
            current_image = self.image.copy()
            height, width = current_image.shape[:2]
            
            # === 차선 변경 중이면 차선 변경 로직만 실행 ===
            if self.update_lane_change():
                # 차선 변경 중이면 현재 차선 상태 퍼블리시하고 다음 루프로
                self.publish_current_lane()
                rate.sleep()
                continue
            
            # 초기화 단계에서 신호등 인식 집중적으로 수행
            if initialization_phase:
                # 신호등 감지 - 초기화 단계에서는 매 프레임 검사
                self.traffic_light_color = self.detect_traffic_light(current_image)
                
                # 차선 감지도 수행
                lane_position, lane_detected, white_lines, yellow_lines, combined_mask = self.get_lane_positions(current_image)
                
                # 시각화
                if self.debug_mode:
                    self.visualize(current_image, lane_position, white_lines, yellow_lines, combined_mask)
                
                # 차량은 정지 상태 유지
                self.drive(0, 0)
                
                initialization_frames += 1
                
                # 일정 프레임 이상 처리했고 빨간색 신호등이 감지되었으면 지연 추가
                if initialization_frames >= required_init_frames:
                    if self.traffic_light_color == "Red" or self.is_red:
                        rospy.loginfo("Red light detected during initialization. Waiting...")
                        time.sleep(0.5)  # 추가 대기
                    
                    rospy.loginfo(f"Initialization complete. Traffic light: {self.traffic_light_color}")
                    rospy.loginfo("Waiting additional 2 seconds for stability...")
                    time.sleep(2)  # 추가 2초 대기
                    initialization_phase = False
                    rospy.loginfo("Now starting normal driving based on traffic light status...")
                
                # 현재 차선 상태 퍼블리시
                self.publish_current_lane()
                rate.sleep()
                continue
            
            # === 정상 주행 단계 (원본 로직 그대로) ===
            # 프레임 처리 빈도 조절
            frame_counter += 1
            if frame_counter % process_every_n != 0:
                rate.sleep()
                continue
            
            # 신호등 감지
            self.traffic_light_color = self.detect_traffic_light(current_image)
            
            # 모든 차선 위치 감지
            lane_position, lane_detected, white_lines, yellow_lines, combined_mask = self.get_lane_positions(current_image)
            
            # 주행 명령 계산
            angle, speed = self.calculate_driving_commands(lane_position, width)
            
            # 신호등이 빨간색이면 반드시 정지
            if self.traffic_light_color == "Red" or self.is_red:
                speed = 0.0
            
            # 주행
            self.drive(angle, speed)
            
            # 현재 차선 상태 퍼블리시
            self.publish_current_lane()
            
            # 시각화
            if self.debug_mode:
                self.visualize(current_image, lane_position, white_lines, yellow_lines, combined_mask)
            
            rate.sleep()

if __name__ == '__main__':
    try:
        detector = LaneDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass