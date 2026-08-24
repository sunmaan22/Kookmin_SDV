#!/usr/bin/env python3
# 라바콘을 피하기 위한 경로 생성 코드

import rospy
import math
import numpy as np
from scipy.interpolate import splprep, splev
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path

class ConeBoundaryBuilder:
    def __init__(self):
        rospy.init_node("cone_boundary_builder")

				# 토픽 subscribe and publish
        self.sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        self.left_pub = rospy.Publisher("/left_boundary", Marker, queue_size=1)
        self.right_pub = rospy.Publisher("/right_boundary", Marker, queue_size=1)
        self.center_pub = rospy.Publisher("/center_path", Marker, queue_size=1)
        self.center_path_pub = rospy.Publisher("/center_path_nav", Path, queue_size=1)

        # 라이다 설정
        self.min_range = 0.2
        self.max_range = 40.0
        self.frame_id = "base_link"

        # lidar to base_link offset (manually set, adjust as needed)
        self.lidar_offset_x = 0  # 10cm behind base_link
        self.lidar_offset_y = 0.0

    # 가까운 점 제거
    def is_close(self, p1, p2, tol=1e-3):
        return abs(p1[0] - p2[0]) < tol and abs(p1[1] - p2[1]) < tol

    # 각도 계산
    def compute_angle(self, v1, v2):
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        norm1 = math.hypot(*v1)
        norm2 = math.hypot(*v2)
        if norm1 == 0 or norm2 == 0:
            return math.pi
        cos_theta = max(min(dot / (norm1 * norm2), 1.0), -1.0)
        return math.acos(cos_theta)

    # 떨어져 있는 라바콘을 이어서 경계선 만들기
    def expand_boundary(self, candidates, start):
        path = [start]
        used = [start]
        prev = None
        curr = start

        while True:
            best_score = float('inf')
            best_point = None
            for pt in candidates:
                if any(self.is_close(pt, u) for u in used):
                    continue
                if pt[0] < curr[0]:
                    continue

                vec_to_pt = (pt[0] - curr[0], pt[1] - curr[1])
                dist = math.hypot(*vec_to_pt)
                if dist > 2.5:
                    continue
                if prev is not None:
                    prev_dir = (curr[0] - prev[0], curr[1] - prev[1])
                    angle = self.compute_angle(prev_dir, vec_to_pt)
                else:
                    angle = 0.0

                score = dist + 3.0 * angle
                if score < best_score:
                    best_score = score
                    best_point = pt

            if best_point is None:
                break
            path.append(best_point)
            used.append(best_point)
            prev = curr
            curr = best_point
        return path

    # 시작 라바콘 감지
    def find_best_initial_pair(self, points, dist_threshold=1.0):
        best_score = -float('inf')
        best_pair = None
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                p1, p2 = points[i], points[j]
                center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                dist_to_robot = math.hypot(center[0], center[1])
                if dist_to_robot > dist_threshold:
                    continue
                y_diff = abs(p1[1] - p2[1])
                x_diff = abs(p1[0] - p2[0])
                if 2 <= y_diff <= 4.5 and x_diff < 2.0:
                    score = y_diff - 0.5 * x_diff
                    if score > best_score:
                        best_score = score
                        left = p1 if p1[1] > p2[1] else p2
                        right = p2 if p1[1] > p2[1] else p1
                        best_pair = (left, right)
        return best_pair

    # 라바콘 중앙 찾기
    def compute_center_path(self, left_path, right_path):
        min_len = min(len(left_path), len(right_path))
        center_path = []
        for i in range(min_len):
            x = (left_path[i][0] + right_path[i][0]) / 2.0
            y = (left_path[i][1] + right_path[i][1]) / 2.0
            center_path.append((x, y))
        return center_path

    # rviz에서 시각화할때 시작점을 (0,0)으로 설정
    def ensure_path_starts_at_origin(self, path):
        """경로가 base_link(0,0)에서 시작하도록 합니다"""
        if len(path) < 1:
            return [(0.0, 0.0)]
            
        # 현재 로봇 위치(base_link)를 경로 시작점으로 추가
        return [(0.0, 0.0)] + path

    # 생성된 노드들을 부드럽게 연결
    def smooth_path(self, path, num_points=None):
        if len(path) < 4:
            return path
        if num_points is None:
            num_points = len(path) * 7

        x = [p[0] for p in path]
        y = [p[1] for p in path]
        tck, _ = splprep([x, y], s=0)
        u_fine = np.linspace(0, 1, num_points)
        x_fine, y_fine = splev(u_fine, tck)
        return list(zip(x_fine, y_fine))

    # 시각화 설정
    def publish_marker(self, path, pub, color):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "cone_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(0.2)
        marker.points = [Point(x=pt[0], y=pt[1], z=0.0) for pt in path]
        pub.publish(marker)

    # 경로 데이터 퍼블리시
    def publish_path(self, path_points):
        rospy.loginfo(f"Publishing path with {len(path_points)} waypoints")
        path_msg = Path()
        path_msg.header.frame_id = "base_link"
        path_msg.header.stamp = rospy.Time.now()
        for pt in path_points:
            pose = PoseStamped()
            pose.header.frame_id = "base_link"
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.center_path_pub.publish(path_msg)

    # 좌표계를 차 기준으로 설정
    def apply_manual_offset(self, path):
        return [(x + self.lidar_offset_x, y + self.lidar_offset_y) for x, y in path]

    # 라이더 콜백 함수
    def scan_callback(self, msg):
        points = []
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        for i, r in enumerate(msg.ranges):
            if self.min_range < r < self.max_range:
                angle = angle_min + i * angle_increment
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                if x > 0:
                    points.append((x, y))

        if len(points) < 6:
            return

        # 1차 시도: 기본 거리 기준(1.0)
        best_pair = self.find_best_initial_pair(points, dist_threshold=1.0)
        if not best_pair:
            rospy.loginfo("No valid cone pair found within 1.0m. Retrying with 3.0m range...")
            # 2차 시도: 완화된 거리 기준(3.0)
            best_pair = self.find_best_initial_pair(points, dist_threshold=3.0)
            if not best_pair:
                rospy.loginfo("No valid cone pair found even with 3.0m range.")
                best_pair = self.find_best_initial_pair(points, dist_threshold=5.0)
                if not best_pair:
                    rospy.loginfo("No valid cone pair found even with 5.0m range.")
                    return

        left_path = self.expand_boundary(points, best_pair[0])
        used_set = left_path
        remaining_points = [pt for pt in points if not any(self.is_close(pt, u) for u in used_set)]
        right_path = self.expand_boundary(remaining_points, best_pair[1])
        center_path = self.compute_center_path(left_path, right_path)
        
        # 경로가 base_link에서 시작하도록 수정
        center_path = self.ensure_path_starts_at_origin(center_path)
        
        smooth_center_path = self.smooth_path(center_path)

        left_path_offset = self.apply_manual_offset(left_path)
        right_path_offset = self.apply_manual_offset(right_path)
        center_path_offset = self.apply_manual_offset(smooth_center_path)
        
        # 최종 생성된 경로 publish
        self.publish_marker(left_path_offset, self.left_pub, (1.0, 0.5, 0.0))
        self.publish_marker(right_path_offset, self.right_pub, (0.0, 0.5, 1.0))
        self.publish_marker(center_path_offset, self.center_pub, (1.0, 1.0, 1.0))
        self.publish_path(center_path_offset)

if __name__ == "__main__":
    try:
        ConeBoundaryBuilder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass