#!/usr/bin/env python3
# PID 제어를 이용하여 생성된 경로를 안정적으로 따라가는 코드

import rospy
import math
import tf
from sensor_msgs.msg import Imu
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from xycar_msgs.msg import XycarMotor
from std_msgs.msg import Bool

class PathFollower:
    def __init__(self):
        rospy.init_node("path_follower")

        self.path = []
        self.yaw = 0.0
        self.prev_target_index = 0
        self.last_path_time = rospy.Time.now()  # path 수신 시간 저장용
        self.turning_override = False

        # PID gains
        self.kp = 350
        self.ki = 0
        self.kd = 1
        self.integral = 0.0
        self.prev_error = 0.0
        self.is_cone_mode = False

        # Subscribers and Publisher
        rospy.Subscriber("/center_path_nav", Path, self.path_callback)
        rospy.Subscriber("/imu", Imu, self.imu_callback)
        self.control_pub = rospy.Publisher("/xycar_motor", XycarMotor, queue_size=1)
        self.cone_mode_pub = rospy.Publisher("/is_cone_mode", Bool, queue_size=1)
        self.last_path_time = rospy.Time.now()

        self.ctrl_timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)  # 20Hz

    # imu 콜백함수
    def imu_callback(self, msg):
        q = msg.orientation
        _, _, self.yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

    # 경로 콜백함수
    def path_callback(self, msg):
        self.path = msg.poses
        self.prev_target_index = 0

        # 새 경로가 수신되면 cone_mode ON
        if not self.is_cone_mode:
            self.is_cone_mode = True
            self.cone_mode_pub.publish(Bool(data=True))
            rospy.loginfo("라바콘 모드 ON")

        rospy.loginfo(f"새 경로 수신: {len(self.path)} 포인트")
        self.last_path_time = rospy.Time.now()

    # 제어 신호
    def send_control(self, speed, steer):
        msg = XycarMotor()
        msg.angle = -steer
        msg.speed = speed
        self.control_pub.publish(msg)

    # 라바콘 모드 종료 함수
    def turn_complete_log(self, event):
        self.is_cone_mode = False
        self.turning_override = False
        self.cone_mode_pub.publish(Bool(data=False)) 
        rospy.loginfo("5초 우회전 완료 → 다음 주행 모드로 자동 전환")

    # 생성된 경로를 주행
    def control_loop(self, event):

        if not self.is_cone_mode:
            return
        
        if self.turning_override:
            return
        
        # ⏱ 경로 수신 시간 검사
        now = rospy.Time.now()
        # 경로가 3초 이상 수신되지 않으면 cone_mode OFF
        if self.is_cone_mode and (now - self.last_path_time).to_sec() > 2:
            self.turning_override = True

            # 5초 동안 우회전 명령
            self.send_control(5.0, -50.0)
            rospy.loginfo("라바콘 모드 OFF → 3초간 우회전 시작")

            # 5초 후 주도권 넘겨줌
            rospy.Timer(rospy.Duration(5.0), self.turn_complete_log, oneshot=True)

            return

        if not self.path:
            return
        
        if self.prev_target_index >= len(self.path) - 2:
            rospy.loginfo("경로 끝에 도달. Pure Pursuit 제어 중단.")
            return

        lookahead = 1.5
        target = None

        # 적합한 목표점 찾기
        for i in range(self.prev_target_index, len(self.path)):
            dx = self.path[i].pose.position.x
            dy = self.path[i].pose.position.y
            dist = math.hypot(dx, dy)
            if dist > lookahead:
                target = self.path[i]
                self.prev_target_index = i
                break

        if target is None:
            # 경로 끝에 가까워졌을 때 마지막 포인트 사용
            if self.path:
                target = self.path[-1]
            else:
                return

        # 차량 기준 좌표계에서 목표점 위치
        dx = target.pose.position.x
        dy = target.pose.position.y
        
        # Pure Pursuit 각도 계산 (baseLink 기준)
        alpha = math.atan2(dy, dx)
        
        # 조향각 계산 (Pure Pursuit 공식)
        L = 0.3  # 차량 축거 (실제 값으로 조정 필요)
        steer_angle = math.atan2(2.0 * L * math.sin(alpha), lookahead)
        
        # PID로 조향 제어
        error = steer_angle
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error

        steer = self.kp * error + self.ki * self.integral + self.kd * derivative
        steer = max(min(steer, 100), -100)

        speed = 5.0
        self.send_control(speed, steer)
        rospy.loginfo(f"[PP] target_idx={self.prev_target_index}, yaw={math.degrees(self.yaw):.1f}, alpha={math.degrees(alpha):.1f}, steer={steer:.2f}")

if __name__ == "__main__":
    try:
        PathFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
