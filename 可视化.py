import time
import datetime
import random
import json
import os
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import streamlit as st
import numpy as np
import pandas as pd
import pytz
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from math import radians, sin, cos, sqrt, asin, pi, atan2, degrees

# ==================== 配置 ====================
AMAP_KEY = "0c475e7a50516001883c104383b43f31"
BEIJING_TZ = pytz.timezone('Asia/Shanghai')
OBSTACLE_FILE = "obstacles.json"

# ==================== 坐标转换 ====================
def wgs84_to_gcj02(lng, lat):
    a = 6378245.0
    ee = 0.00669342162296594323
    def transform_lat(x, y):
        ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * abs(x)
        ret += (20.0 * sin(6.0 * x * pi) + 20.0 * sin(2.0 * x * pi)) * 2.0 / 3.0
        ret += (20.0 * sin(y * pi) + 40.0 * sin(y / 3.0 * pi)) * 2.0 / 3.0
        ret += (160.0 * sin(y / 12.0 * pi) + 320 * sin(y * pi / 30.0)) * 2.0 / 3.0
        return ret
    def transform_lng(x, y):
        ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * abs(x)
        ret += (20.0 * sin(6.0 * x * pi) + 20.0 * sin(2.0 * x * pi)) * 2.0 / 3.0
        ret += (20.0 * sin(x * pi) + 40.0 * sin(x / 3.0 * pi)) * 2.0 / 3.0
        ret += (150.0 * sin(x / 12.0 * pi) + 300.0 * sin(x / 30.0 * pi)) * 2.0 / 3.0
        return ret
    dlat = transform_lat(lng - 105.0, lat - 35.0)
    dlng = transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return mglng, mglat

def haversine(lon1, lat1, lon2, lat2):
    R = 6371
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

# ==================== 几何工具 ====================
def on_segment(p, q, r):
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
        q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False

def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0: return 0
    return 1 if val > 0 else 2

def segments_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True
    return False

def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i+1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2-x1)*(y-y1)/(y2-y1) + x1):
            inside = not inside
    return inside

def line_polygon_intersect(line_start, line_end, polygon):
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i+1) % len(polygon)]
        if segments_intersect(line_start, line_end, p1, p2):
            return True
    if point_in_polygon(line_start, polygon) or point_in_polygon(line_end, polygon):
        return True
    return False

def get_polygon_center(polygon):
    lng_sum = sum(p[0] for p in polygon)
    lat_sum = sum(p[1] for p in polygon)
    return lng_sum/len(polygon), lat_sum/len(polygon)

def point_to_polygon_min_distance(point, polygon):
    min_dist = float('inf')
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i+1) % len(polygon)]
        x0, y0 = point
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            dist = haversine(x0, y0, x1, y1)
        else:
            t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx*dx + dy*dy)
            t = max(0, min(1, t))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist = haversine(x0, y0, proj_x, proj_y)
        if dist < min_dist:
            min_dist = dist
    return min_dist

def ensure_point_safe_from_obstacles(point, obstacle, safe_dist_km, max_iter=10):
    current = point
    for _ in range(max_iter):
        min_d = point_to_polygon_min_distance(current, obstacle['coordinates'])
        if min_d >= safe_dist_km - 1e-6:
            return current, True
        center = get_polygon_center(obstacle['coordinates'])
        dx = current[0] - center[0]
        dy = current[1] - center[1]
        length = sqrt(dx*dx + dy*dy)
        if length < 1e-9:
            dx, dy = 1.0, 0.0
            length = 1.0
        need = safe_dist_km - min_d
        step_deg = (need * 1.2) / 111.0
        new_x = current[0] + (dx / length) * step_deg
        new_y = current[1] + (dy / length) * step_deg
        current = (new_x, new_y)
    return current, False

def ensure_point_safe_from_all_obstacles(point, obstacles, flight_altitude, safe_dist_km):
    result = point
    all_safe = True
    for obs in obstacles:
        if obs.get('height', 50) >= flight_altitude:
            result, safe = ensure_point_safe_from_obstacles(result, obs, safe_dist_km)
            if not safe:
                all_safe = False
    return result, all_safe

# ==================== 多路径避障算法 ====================
def point_side_of_line(point, line_start, line_end):
    return (line_end[0] - line_start[0]) * (point[1] - line_start[1]) - (line_end[1] - line_start[1]) * (point[0] - line_start[0])

def get_side_waypoints(polygon, start, end, safe_dist_km, side='left', obstacles=None, flight_altitude=50):
    center = get_polygon_center(polygon)
    best_vertex = None
    best_side_val = None
    for v in polygon:
        side_val = point_side_of_line(v, start, end)
        if side == 'left' and side_val > 0:
            if best_side_val is None or side_val > best_side_val:
                best_side_val = side_val
                best_vertex = v
        elif side == 'right' and side_val < 0:
            if best_side_val is None or side_val < best_side_val:
                best_side_val = side_val
                best_vertex = v
    if best_vertex is None:
        best_vertex = min(polygon, key=lambda p: haversine(p[0], p[1], center[0], center[1]))
    dx = best_vertex[0] - center[0]
    dy = best_vertex[1] - center[1]
    length = sqrt(dx*dx + dy*dy)
    if length < 1e-9:
        dx, dy = 1.0, 0.0
        length = 1.0
    delta_deg = safe_dist_km / 111.0
    wp = (best_vertex[0] + (dx / length) * delta_deg,
          best_vertex[1] + (dy / length) * delta_deg)
    wp, _ = ensure_point_safe_from_obstacles(wp, {'coordinates': polygon}, safe_dist_km)
    if obstacles is not None:
        wp, _ = ensure_point_safe_from_all_obstacles(wp, obstacles, flight_altitude, safe_dist_km)
    if point_in_polygon(wp, polygon):
        wp = (wp[0] + (dx / length) * delta_deg,
              wp[1] + (dy / length) * delta_deg)
        wp, _ = ensure_point_safe_from_obstacles(wp, {'coordinates': polygon}, safe_dist_km)
    return wp

def find_path_with_side(start, end, obstacles, flight_altitude, safe_dist_km, side, depth=0):
    MAX_DEPTH = 10
    if depth > MAX_DEPTH:
        return [(start, end)], haversine(start[0], start[1], end[0], end[1])
    blocking = []
    for obs in obstacles:
        if obs.get('height', 50) >= flight_altitude:
            poly = obs['coordinates']
            if line_polygon_intersect(start, end, poly):
                blocking.append(obs)
    if not blocking:
        return [(start, end)], haversine(start[0], start[1], end[0], end[1])
    obs = blocking[0]
    poly = obs['coordinates']
    if side == 'optimal':
        left_wp = get_side_waypoints(poly, start, end, safe_dist_km, 'left', obstacles, flight_altitude)
        right_wp = get_side_waypoints(poly, start, end, safe_dist_km, 'right', obstacles, flight_altitude)
        left_segs, left_dist = find_path_with_side(start, left_wp, obstacles, flight_altitude, safe_dist_km, 'optimal', depth+1)
        right_segs, right_dist = find_path_with_side(start, right_wp, obstacles, flight_altitude, safe_dist_km, 'optimal', depth+1)
        left_segs2, left_dist2 = find_path_with_side(left_wp, end, obstacles, flight_altitude, safe_dist_km, 'optimal', depth+1)
        right_segs2, right_dist2 = find_path_with_side(right_wp, end, obstacles, flight_altitude, safe_dist_km, 'optimal', depth+1)
        left_total = left_dist + left_dist2
        right_total = right_dist + right_dist2
        if left_total < right_total:
            return left_segs + left_segs2, left_total
        else:
            return right_segs + right_segs2, right_total
    else:
        wp = get_side_waypoints(poly, start, end, safe_dist_km, side, obstacles, flight_altitude)
        left_segs, left_dist = find_path_with_side(start, wp, obstacles, flight_altitude, safe_dist_km, side, depth+1)
        right_segs, right_dist = find_path_with_side(wp, end, obstacles, flight_altitude, safe_dist_km, side, depth+1)
        return left_segs + right_segs, left_dist + right_dist

# ==================== 曲线平滑 ====================
def bezier_curve(points, num_points=50):
    if len(points) < 2:
        return points
    smoothed = []
    for i in range(len(points)-1):
        p0 = points[i]
        p3 = points[i+1]
        if i == 0:
            p1 = (p0[0] + (p3[0]-p0[0])*0.25, p0[1] + (p3[1]-p0[1])*0.25)
        else:
            p1 = (p0[0] + (p3[0]-points[i-1][0])*0.2, p0[1] + (p3[1]-points[i-1][1])*0.2)
        if i == len(points)-2:
            p2 = (p3[0] - (p3[0]-p0[0])*0.25, p3[1] - (p3[1]-p0[1])*0.25)
        else:
            p2 = (p3[0] - (points[i+2][0]-p0[0])*0.2, p3[1] - (points[i+2][1]-p0[1])*0.2)
        for t in np.linspace(0, 1, num_points//(len(points)-1)):
            x = (1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0]
            y = (1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1]
            smoothed.append((x, y))
    smoothed.append(points[-1])
    unique = []
    for p in smoothed:
        if not unique or haversine(p[0], p[1], unique[-1][0], unique[-1][1]) > 1e-6:
            unique.append(p)
    return unique

def get_perpendicular_hover_point(end_point, approach_dir, distance_m=10.0, side='right'):
    ux, uy = approach_dir
    if side == 'left':
        nx = -uy
        ny = ux
    else:
        nx = uy
        ny = -ux
    dist_deg = distance_m / 1000.0 / 111.0
    lon = end_point[0] + nx * dist_deg
    lat = end_point[1] + ny * dist_deg
    candidate = (lon, lat)
    actual_dist = haversine(end_point[0], end_point[1], candidate[0], candidate[1]) * 1000
    if abs(actual_dist - distance_m) > 0.1:
        scale = distance_m / actual_dist
        lon = end_point[0] + nx * dist_deg * scale
        lat = end_point[1] + ny * dist_deg * scale
        candidate = (lon, lat)
    return candidate

def get_safe_hover_point(end_point, approach_dir, obstacles, flight_altitude, distance_m=10.0, safe_dist_km=0.01):
    candidates = []
    for side in ['left', 'right']:
        pt = get_perpendicular_hover_point(end_point, approach_dir, distance_m, side)
        safe = True
        for obs in obstacles:
            if obs.get('height', 50) >= flight_altitude:
                if point_in_polygon(pt, obs['coordinates']):
                    safe = False
                    break
                min_d = point_to_polygon_min_distance(pt, obs['coordinates'])
                if min_d < safe_dist_km - 1e-6:
                    safe = False
                    break
        def point_to_segment_distance(p, a, b):
            x0,y0=p; x1,y1=a; x2,y2=b
            dx=x2-x1; dy=y2-y1
            if dx==0 and dy==0:
                return haversine(x0,y0,x1,y1)
            t = ((x0-x1)*dx + (y0-y1)*dy) / (dx*dx+dy*dy)
            t = max(0, min(1, t))
            proj_x = x1 + t*dx
            proj_y = y1 + t*dy
            return haversine(x0,y0,proj_x,proj_y)
        line_start = (end_point[0] - approach_dir[0]*0.01, end_point[1] - approach_dir[1]*0.01)
        line_end = end_point
        d = point_to_segment_distance(pt, line_start, line_end)
        if d < 0.001:
            safe = False
        if safe:
            candidates.append((pt, side))
    if candidates:
        pt = candidates[0][0]
        pt, _ = ensure_point_safe_from_all_obstacles(pt, obstacles, flight_altitude, safe_dist_km)
        return pt, True
    else:
        back_pt = (end_point[0] - approach_dir[0] * distance_m/1000/111.0,
                   end_point[1] - approach_dir[1] * distance_m/1000/111.0)
        actual = haversine(end_point[0], end_point[1], back_pt[0], back_pt[1])*1000
        if abs(actual - distance_m) > 0.1:
            scale = distance_m / actual
            back_pt = (end_point[0] - approach_dir[0] * distance_m/1000/111.0 * scale,
                       end_point[1] - approach_dir[1] * distance_m/1000/111.0 * scale)
        back_pt, _ = ensure_point_safe_from_all_obstacles(back_pt, obstacles, flight_altitude, safe_dist_km)
        return back_pt, False

def check_landing_safety(destination, obstacles, flight_altitude, safe_radius_km=0.01):
    min_dist = float('inf')
    nearest_obs = None
    for obs in obstacles:
        if obs.get('height', 50) >= flight_altitude:
            poly = obs['coordinates']
            dist = point_to_polygon_min_distance(destination, poly)
            if dist < min_dist:
                min_dist = dist
                nearest_obs = obs.get('name', '未知障碍物')
    if min_dist < safe_radius_km:
        return False, min_dist, nearest_obs
    return True, min_dist, None

# ==================== 飞行监控模拟器 ====================
class FlightSimulator:
    def __init__(self, waypoints, speed_mps=8.5, battery_reserve_percent=80):
        self.waypoints = waypoints
        self.speed = speed_mps
        self.total_distance = sum(haversine(waypoints[i][0], waypoints[i][1], waypoints[i+1][0], waypoints[i+1][1]) 
                                   for i in range(len(waypoints)-1)) * 1000
        self.dist_traveled = 0.0
        self.current_index = 0
        self.current_pos = waypoints[0] if waypoints else None
        self.battery_reserve_percent = battery_reserve_percent
        self.start_abs_time = None
        self.pause_start_time = None
        self.total_paused_duration = 0.0
        self.is_running = False
        self.is_paused = False
        self.battery_percent = 100.0

    @property
    def elapsed_seconds(self):
        if not self.is_running and not self.is_paused:
            return 0.0
        if self.is_running:
            if self.start_abs_time is None:
                return 0.0
            return time.time() - self.start_abs_time
        else:
            return self.total_paused_duration

    def get_elapsed_time(self):
        return self.elapsed_seconds

    def start(self):
        if not self.is_running and not self.is_paused:
            self.start_abs_time = time.time()
            self.total_paused_duration = 0.0
            self.is_running = True
            self.is_paused = False
            self.dist_traveled = 0.0
            self.current_index = 0
            self.current_pos = self.waypoints[0]

    def pause(self):
        if self.is_running and not self.is_paused:
            self.pause_start_time = time.time()
            self.is_paused = True
            self.is_running = False

    def resume(self):
        if not self.is_running and self.is_paused:
            self.total_paused_duration += time.time() - self.pause_start_time
            self.start_abs_time = time.time() - self.total_paused_duration
            self.is_paused = False
            self.is_running = True

    def stop(self):
        self.is_running = False
        self.is_paused = False
        self.start_abs_time = None
        self.pause_start_time = None
        self.total_paused_duration = 0.0
        self.dist_traveled = 0.0
        self.current_index = 0
        self.current_pos = self.waypoints[0] if self.waypoints else None
        self.battery_percent = 100.0

    def update(self):
        if not self.is_running or self.is_paused:
            return
        if self.start_abs_time is None:
            return
        elapsed = time.time() - self.start_abs_time
        target_dist = self.speed * elapsed
        if target_dist >= self.total_distance:
            self.current_index = len(self.waypoints) - 1
            self.current_pos = self.waypoints[-1]
            self.dist_traveled = self.total_distance
            self.is_running = False
            self.start_abs_time = None
        else:
            dist_accum = 0.0
            for i in range(len(self.waypoints)-1):
                seg_dist = haversine(self.waypoints[i][0], self.waypoints[i][1], 
                                     self.waypoints[i+1][0], self.waypoints[i+1][1]) * 1000
                if target_dist <= dist_accum + seg_dist:
                    t = (target_dist - dist_accum) / seg_dist if seg_dist > 0 else 0
                    lon = self.waypoints[i][0] + t * (self.waypoints[i+1][0] - self.waypoints[i][0])
                    lat = self.waypoints[i][1] + t * (self.waypoints[i+1][1] - self.waypoints[i][1])
                    self.current_pos = (lon, lat)
                    self.current_index = i + 1
                    self.dist_traveled = target_dist
                    break
                dist_accum += seg_dist
        progress = self.dist_traveled / self.total_distance if self.total_distance > 0 else 1
        self.battery_percent = max(0, 100 - progress * (100 - self.battery_reserve_percent))

# ==================== 心跳模拟器 ====================
class DroneHeartbeatSimulator:
    def __init__(self, timeout_seconds=3):
        self.timeout_seconds = timeout_seconds
        self.sequence_number = 0
        self.heartbeat_history = deque(maxlen=100)
        self.timeout_events = deque(maxlen=20)
        self.last_received_time = time.time()
        self.start_time = time.time()
        self.total_sent = 0
        self.total_lost = 0
        self.last_timeout_time = 0

    def get_beijing_time(self):
        return datetime.datetime.now(BEIJING_TZ)
    
    def generate_heartbeat(self):
        timestamp = self.get_beijing_time()
        self.total_sent += 1
        if random.random() < 0.1:
            self.total_lost += 1
            self._check_timeout()
            return None
        delay_ms = random.uniform(100, 500)
        receive_time = self.get_beijing_time()
        record = {
            'sequence': self.sequence_number,
            'send_time': timestamp,
            'receive_time': receive_time,
            'delay_ms': delay_ms,
            'status': 'received'
        }
        self.heartbeat_history.append(record)
        self.last_received_time = time.time()
        self.sequence_number += 1
        self._check_timeout()
        return record
    
    def _check_timeout(self):
        current_time = time.time()
        if current_time - self.last_received_time > self.timeout_seconds:
            if current_time - self.last_timeout_time > 1:
                self.timeout_events.append({
                    'time': self.get_beijing_time(),
                    'duration': current_time - self.last_received_time
                })
                self.last_timeout_time = current_time
    
    def get_recent_data(self, window_size=30):
        sequences, delays, receive_times = [], [], []
        if not self.heartbeat_history:
            return sequences, delays, receive_times
        for record in self.heartbeat_history:
            if record and isinstance(record, dict):
                sequences.append(record.get('sequence', 0))
                delays.append(record.get('delay_ms', 0))
                receive_times.append(record.get('receive_time', datetime.datetime.now(BEIJING_TZ)))
        if len(sequences) > window_size:
            sequences = sequences[-window_size:]
            delays = delays[-window_size:]
            receive_times = receive_times[-window_size:]
        return sequences, delays, receive_times
    
    def get_statistics(self):
        if not self.heartbeat_history:
            return {'avg_delay': 0, 'min_delay': 0, 'max_delay': 0, 'packet_loss_rate': 0, 'received_count': 0}
        delays = [r['delay_ms'] for r in self.heartbeat_history if isinstance(r, dict) and 'delay_ms' in r]
        if not delays:
            return {'avg_delay': 0, 'min_delay': 0, 'max_delay': 0, 'packet_loss_rate': 0, 'received_count': len(self.heartbeat_history)}
        packet_loss_rate = (self.total_lost / self.total_sent * 100) if self.total_sent > 0 else 0
        return {
            'avg_delay': sum(delays) / len(delays),
            'min_delay': min(delays),
            'max_delay': max(delays),
            'packet_loss_rate': packet_loss_rate,
            'received_count': len(self.heartbeat_history)
        }

# ==================== 辅助函数 ====================
def get_beijing_time_info():
    now = datetime.datetime.now(BEIJING_TZ)
    return {
        'datetime': now,
        'time_str': now.strftime('%Y年%m月%d日 %H:%M:%S'),
        'weekday': now.strftime('%A'),
        'timezone': 'Asia/Shanghai (UTC+8)'
    }

def format_beijing_time(dt):
    if dt is None:
        return "N/A"
    if dt.tzinfo is None:
        dt = BEIJING_TZ.localize(dt)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def create_heartbeat_charts(sequences, delays, receive_times, timeout_count, timeout_events):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    if sequences and delays and receive_times and len(sequences) > 0:
        try:
            ax1.plot(receive_times, delays, 'b-o', markersize=6, linewidth=2)
            ax1.set_xlabel('接收时间（北京时间）', fontsize=12, fontweight='bold')
            ax1.set_ylabel('延迟 (ms)', fontsize=12, fontweight='bold')
            ax1.set_title('实时心跳延迟监控（按北京时间）', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            if delays:
                avg_delay = sum(delays) / len(delays)
                ax1.axhline(y=avg_delay, color='r', linestyle='--', linewidth=2, label=f'平均延迟: {avg_delay:.1f}ms')
                ax1.legend(loc='upper right')
            ax1.axhline(y=400, color='orange', linestyle=':', linewidth=1.5, label='延迟阈值: 400ms', alpha=0.7)
            threshold = 400
            above_threshold = [d if d > threshold else threshold for d in delays]
            ax1.fill_between(receive_times, threshold, above_threshold, alpha=0.3, color='red', label='超出阈值')
            ax2.plot(receive_times, sequences, 'g-o', markersize=6, linewidth=2)
            ax2.set_xlabel('接收时间（北京时间）', fontsize=12, fontweight='bold')
            ax2.set_ylabel('心跳序号', fontsize=12, fontweight='bold')
            ax2.set_title(f'心跳序号接收情况 | 超时次数: {timeout_count}', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            if timeout_events and len(timeout_events) > 0:
                now_beijing = datetime.datetime.now(BEIJING_TZ)
                recent_timeouts = [e for e in timeout_events if e and isinstance(e, dict) and 'time' in e and (now_beijing - e['time']).total_seconds() < 10]
                if recent_timeouts:
                    ax2.text(0.02, 0.98, f"⚠️ 最近超时: {len(recent_timeouts)}次", transform=ax2.transAxes, fontsize=11,
                            verticalalignment='top', fontweight='bold', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        except Exception as e:
            ax1.text(0.5, 0.5, f'绘图错误: {str(e)}', ha='center', va='center')
            ax2.text(0.5, 0.5, '请检查数据', ha='center', va='center')
    else:
        ax1.text(0.5, 0.5, '等待数据...', ha='center', va='center', fontsize=14)
        ax2.text(0.5, 0.5, '等待数据...', ha='center', va='center', fontsize=14)
        ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
        ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
    plt.tight_layout()
    return fig

# ==================== 障碍物管理 ====================
def load_obstacles():
    if os.path.exists(OBSTACLE_FILE):
        try:
            with open(OBSTACLE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_obstacles(obstacles):
    with open(OBSTACLE_FILE, 'w', encoding='utf-8') as f:
        json.dump(obstacles, f, ensure_ascii=False, indent=2)

# ==================== 通信链路拓扑HTML组件 ====================
def link_topology_html(delay_ms, loss_rate):
    return f"""
    <div style="background: #1e1e2f; border-radius: 12px; padding: 15px; color: white; font-family: monospace;">
        <div style="display: flex; justify-content: space-around; align-items: center; margin-bottom: 20px;">
            <div style="text-align: center;">
                <div style="font-size: 24px;">🖥️</div>
                <div><b>GCS</b></div>
                <div style="font-size: 12px; color: #aaa;">地面站</div>
                <div style="font-size: 11px;">192.168.1.100</div>
                <div style="color: #4CAF50;">● 已连接</div>
            </div>
            <div style="font-size: 20px;">→</div>
            <div style="text-align: center;">
                <div style="font-size: 24px;">💻</div>
                <div><b>OBC</b></div>
                <div style="font-size: 12px; color: #aaa;">机载计算机</div>
                <div style="font-size: 11px;">Raspberry Pi 4</div>
                <div style="color: #4CAF50;">● 已连接</div>
            </div>
            <div style="font-size: 20px;">→</div>
            <div style="text-align: center;">
                <div style="font-size: 24px;">🛸</div>
                <div><b>FCU</b></div>
                <div style="font-size: 12px; color: #aaa;">飞控</div>
                <div style="font-size: 11px;">PX4/ArduPilot</div>
                <div style="color: #4CAF50;">● 已连接</div>
            </div>
        </div>
        <div style="background: #2a2a3a; border-radius: 8px; padding: 10px; margin-top: 10px;">
            <div style="display: flex; justify-content: space-between;">
                <span><b>链路统计</b></span>
                <span>📡 UDP:14550</span>
                <span>⚡ MAVLink</span>
            </div>
            <hr style="border-color: #444;">
            <div>📶 GCS ↔ OBC : <span style="color:#4CAF50;">正常</span></div>
            <div>📶 OBC ↔ FCU : <span style="color:#4CAF50;">正常</span></div>
            <div>⏱️ 延迟: ~{delay_ms:.0f} ms</div>
            <div>📉 丢包率: {loss_rate:.1f}%</div>
        </div>
    </div>
    """

# ==================== Streamlit 界面 ====================
st.set_page_config(page_title="无人机监控与智能航线规划", layout="wide", page_icon="🚁")

st.markdown("""
<style>
    .time-display { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 20px; }
    .beijing-badge { background-color: #ff6b6b; color: white; padding: 5px 10px; border-radius: 5px; font-size: 12px; font-weight: bold; display: inline-block; margin-left: 10px; }
    .warning-text { color: #ff4b4b; font-weight: bold; animation: blink 1s infinite; }
    @keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    .status-badge { padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; display: inline-block; }
    .status-set { background-color: #4CAF50; color: white; }
    .status-notset { background-color: #f44336; color: white; }
    .safe-text { color: #4CAF50; font-weight: bold; }
    .danger-text { color: #f44336; font-weight: bold; }
    .warning-text-yellow { color: #ff9800; font-weight: bold; }
    .info-text { color: #2196F3; font-weight: bold; }
    .monitor-card { background: #f0f2f6; border-radius: 10px; padding: 15px; margin: 10px 0; text-align: center; }
</style>
""", unsafe_allow_html=True)

# 初始化 session_state
if "simulator" not in st.session_state:
    st.session_state.simulator = DroneHeartbeatSimulator(timeout_seconds=3)
if "running" not in st.session_state:
    st.session_state.running = False
if "last_update" not in st.session_state:
    st.session_state.last_update = time.time()
if "map_points" not in st.session_state:
    st.session_state.map_points = []
if "a_point" not in st.session_state:
    st.session_state.a_point = None
if "b_point" not in st.session_state:
    st.session_state.b_point = None
if "input_coordinate_system" not in st.session_state:
    st.session_state.input_coordinate_system = "WGS-84"
if "page" not in st.session_state:
    st.session_state.page = "心跳监控"
if "obstacles" not in st.session_state:
    st.session_state.obstacles = load_obstacles()
if "flight_altitude" not in st.session_state:
    st.session_state.flight_altitude = 100.0
if "map_style" not in st.session_state:
    st.session_state.map_style = "卫星影像"
if "avoidance_enabled" not in st.session_state:
    st.session_state.avoidance_enabled = True
if "safe_distance" not in st.session_state:
    st.session_state.safe_distance = 0.05
if "route_side" not in st.session_state:
    st.session_state.route_side = "最优路径"
if "curve_smooth" not in st.session_state:
    st.session_state.curve_smooth = False
if "landing_safety" not in st.session_state:
    st.session_state.landing_safety = True
if "flight_speed" not in st.session_state:
    st.session_state.flight_speed = 8.5
if "flight_sim" not in st.session_state:
    st.session_state.flight_sim = None
if "planned_waypoints" not in st.session_state:
    st.session_state.planned_waypoints = []
if "battery_reserve_percent" not in st.session_state:
    st.session_state.battery_reserve_percent = 80
# ===== 新增：飞行事件日志存储 =====
if "flight_events" not in st.session_state:
    st.session_state.flight_events = []
if "last_flight_status" not in st.session_state:
    st.session_state.last_flight_status = None
if "arrived_logged" not in st.session_state:
    st.session_state.arrived_logged = False

st.title("🚁 无人机实时监控与智能航线规划系统")
st.markdown('<span class="beijing-badge">🇨🇳 北京时间 (UTC+8)</span>', unsafe_allow_html=True)
current_time_info = get_beijing_time_info()
st.markdown(f"""
<div class="time-display">
    🕐 {current_time_info['time_str']}<br>
    <div class="time-sub">📍 {current_time_info['weekday']} | 时区: {current_time_info['timezone']}</div>
</div>
""", unsafe_allow_html=True)

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("⚙️ 全局控制")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶ 开始心跳", use_container_width=True, type="primary"):
            st.session_state.running = True
    with col2:
        if st.button("⏸ 停止心跳", use_container_width=True):
            st.session_state.running = False
    with col3:
        if st.button("🔄 重置心跳", use_container_width=True):
            st.session_state.simulator = DroneHeartbeatSimulator(timeout_seconds=3)
            st.session_state.running = False
            st.session_state.last_update = time.time()
    st.divider()
    st.subheader("📊 心跳统计")
    sim = st.session_state.simulator
    stats = sim.get_statistics()
    st.metric("成功接收", stats['received_count'])
    st.metric("超时事件", len(sim.timeout_events))
    st.metric("平均延迟", f"{stats['avg_delay']:.1f} ms")
    st.metric("丢包率", f"{stats['packet_loss_rate']:.1f}%")
    runtime = time.time() - sim.start_time
    st.metric("心跳时长", f"{int(runtime // 60)}分{int(runtime % 60)}秒")
    
    st.divider()
    st.subheader("📌 系统状态")
    a_status = "已设" if st.session_state.a_point else "未设"
    b_status = "已设" if st.session_state.b_point else "未设"
    st.markdown(f"**A点** : <span class='status-badge {'status-set' if st.session_state.a_point else 'status-notset'}'>{a_status}</span>", unsafe_allow_html=True)
    st.markdown(f"**B点** : <span class='status-badge {'status-set' if st.session_state.b_point else 'status-notset'}'>{b_status}</span>", unsafe_allow_html=True)
    if st.session_state.a_point:
        st.caption(f"原始: {st.session_state.a_point['original_lat']:.6f}, {st.session_state.a_point['original_lon']:.6f} ({st.session_state.a_point['original_crs']})")
    if st.session_state.b_point:
        st.caption(f"原始: {st.session_state.b_point['original_lat']:.6f}, {st.session_state.b_point['original_lon']:.6f} ({st.session_state.b_point['original_crs']})")
    st.metric("🚁 巡航高度", f"{st.session_state.flight_altitude:.0f} m")
    st.divider()
    st.subheader("⚡ 刷新设置")
    refresh_rate = st.selectbox("刷新频率（秒）", [1, 2, 3, 5], index=0)
    st.divider()
    st.subheader("🧭 功能页面")
    page = st.radio("跳转", ["心跳监控", "任务执行", "航线规划", "障碍物管理", "坐标系设置"], 
                    index=["心跳监控", "任务执行", "航线规划", "障碍物管理", "坐标系设置"].index(st.session_state.page))
    st.session_state.page = page

# ==================== 自动心跳生成 ====================
if st.session_state.running:
    current_time = time.time()
    if current_time - st.session_state.last_update >= refresh_rate:
        record = st.session_state.simulator.generate_heartbeat()
        st.session_state.last_update = current_time
        if record:
            st.toast(f"✅ 心跳 #{record['sequence']} | 延迟: {record['delay_ms']:.1f}ms", icon="✅")
        else:
            st.toast(f"⚠️ 心跳丢失", icon="⚠️")

# ==================== 页面1：心跳监控 ====================
if st.session_state.page == "心跳监控":
    st.header("📡 心跳监控 · 实时心跳数据")
    sim = st.session_state.simulator
    stats = sim.get_statistics()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📊 成功接收", stats['received_count'])
    c2.metric("⚠️ 超时事件", len(sim.timeout_events))
    c3.metric("⏱️ 平均延迟", f"{stats['avg_delay']:.1f} ms", delta=f"{stats['min_delay']:.0f}-{stats['max_delay']:.0f}ms")
    c4.metric("📉 丢包率", f"{stats['packet_loss_rate']:.1f}%")
    c5.metric("⏰ 运行时长", f"{int((time.time()-sim.start_time)//60)}分{int((time.time()-sim.start_time)%60)}秒")
    st.markdown("---")
    try:
        seq, delay, rtimes = sim.get_recent_data(30)
        fig = create_heartbeat_charts(seq, delay, rtimes, len(sim.timeout_events), sim.timeout_events)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"图表错误: {e}")
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("📡 最新心跳信息")
        if sim.heartbeat_history:
            latest = sim.heartbeat_history[-1]
            st.markdown(f"- **序号**: {latest.get('sequence', 'N/A')}")
            st.markdown(f"- **延迟**: {latest.get('delay_ms', 0):.1f} ms")
            st.markdown(f"- **接收时间**: {format_beijing_time(latest.get('receive_time'))}")
            delay_val = latest.get('delay_ms', 0)
            if delay_val < 200:
                st.success("✅ 延迟状态: 优秀 (<200ms)")
            elif delay_val < 400:
                st.warning("⚠️ 延迟状态: 良好 (200-400ms)")
            else:
                st.error("🔴 延迟状态: 较差 (>400ms)")
        else:
            st.info("等待数据...")
    with col_right:
        st.subheader("⚠️ 最近超时事件")
        if sim.timeout_events:
            df_timeout = pd.DataFrame([{
                "时间": e['time'].strftime('%H:%M:%S'),
                "持续": f"{e['duration']:.1f}秒"
            } for e in list(sim.timeout_events)[-5:] if e and isinstance(e, dict)])
            st.dataframe(df_timeout, use_container_width=True)
            now = sim.get_beijing_time()
            if any((now - e['time']).total_seconds() < 10 for e in sim.timeout_events if e and 'time' in e):
                st.markdown('<p class="warning-text">⚠️ 最近10秒内有超时发生！</p>', unsafe_allow_html=True)
        else:
            st.success("✅ 无超时事件")
    st.subheader("📊 传输统计")
    if sim.heartbeat_history:
        delays_hist = [r['delay_ms'] for r in list(sim.heartbeat_history)[-50:] if isinstance(r, dict) and 'delay_ms' in r]
        if delays_hist:
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.hist(delays_hist, bins=20, color='skyblue', edgecolor='black')
            ax1.axvline(x=400, color='red', linestyle='--', label='阈值400ms')
            ax1.set_xlabel('延迟 (ms)'); ax1.set_ylabel('频次'); ax1.set_title('延迟分布'); ax1.legend()
            ax2.plot(rtimes[-50:], delays_hist, 'b-', alpha=0.7)
            ax2.scatter(rtimes[-50:], delays_hist, c='red', s=30, alpha=0.5)
            ax2.set_xlabel('接收时间（北京时间）'); ax2.set_ylabel('延迟 (ms)'); ax2.set_title('延迟变化趋势')
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

    # ===== 新增：心跳详细日志（可展开，可下载CSV） =====
    st.markdown("---")
    with st.expander("📋 查看完整心跳日志（最近100条）", expanded=False):
        history = list(sim.heartbeat_history)
        if history:
            df_log = pd.DataFrame(history)
            if 'receive_time' in df_log.columns:
                df_log['receive_time'] = df_log['receive_time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] if x else '')
            if 'send_time' in df_log.columns:
                df_log['send_time'] = df_log['send_time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] if x else '')
            df_display = df_log[['sequence', 'receive_time', 'delay_ms', 'status']].sort_values('receive_time', ascending=False)
            st.dataframe(df_display, use_container_width=True, height=300)
            csv = df_log.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 下载心跳日志 CSV",
                data=csv,
                file_name=f"heartbeat_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.info("暂无心跳记录，请先点击「开始心跳」")

# ==================== 页面2：任务执行 ====================
elif st.session_state.page == "任务执行":
    st.header("✈️ 飞行实时画面 - 任务执行监控")
    
    # 电量调节
    col_reserve1, col_reserve2 = st.columns([1, 3])
    with col_reserve1:
        new_reserve = st.slider("🔋 到达终点时剩余电量 (%)", 0, 100, st.session_state.battery_reserve_percent, 5,
                                help="数值越高，电量消耗越慢。例如80%表示全程仅消耗20%电量。修改后需停止并重新开始任务。")
        if new_reserve != st.session_state.battery_reserve_percent:
            st.session_state.battery_reserve_percent = new_reserve
            if st.session_state.flight_sim is not None and not st.session_state.flight_sim.is_running and not st.session_state.flight_sim.is_paused:
                if st.session_state.planned_waypoints:
                    st.session_state.flight_sim = FlightSimulator(
                        st.session_state.planned_waypoints,
                        speed_mps=st.session_state.flight_speed,
                        battery_reserve_percent=st.session_state.battery_reserve_percent
                    )
                    st.info("电量参数已更新，请重新开始任务")
    with col_reserve2:
        st.caption("修改电量保留率后，需要停止并重新开始飞行任务才能生效。")
    
    if not st.session_state.planned_waypoints:
        if st.session_state.a_point and st.session_state.b_point:
            start_pt = (st.session_state.a_point['lon_gcj'], st.session_state.a_point['lat_gcj'])
            end_pt = (st.session_state.b_point['lon_gcj'], st.session_state.b_point['lat_gcj'])
            st.session_state.planned_waypoints = [start_pt, end_pt]
            st.info("⚠️ 未找到规划航线，已使用A/B点直线作为临时航线。请前往「航线规划」页面生成最优避障路径。")
        else:
            st.warning("⚠️ 请先在「航线规划」页面设置A点和B点并生成航线。")
            st.stop()
    
    if (st.session_state.flight_sim is None or 
        st.session_state.flight_sim.waypoints != st.session_state.planned_waypoints or
        st.session_state.flight_sim.battery_reserve_percent != st.session_state.battery_reserve_percent):
        st.session_state.flight_sim = FlightSimulator(
            st.session_state.planned_waypoints,
            speed_mps=st.session_state.flight_speed,
            battery_reserve_percent=st.session_state.battery_reserve_percent
        )
    
    sim = st.session_state.flight_sim
    
    # 控制按钮
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    with col_btn1:
        if st.button("▶ 开始任务", use_container_width=True, type="primary"):
            if not sim.is_running and not sim.is_paused:
                sim.start()
            elif sim.is_paused:
                sim.resume()
    with col_btn2:
        if st.button("⏸ 暂停", use_container_width=True):
            if sim.is_running:
                sim.pause()
    with col_btn3:
        if st.button("⏹️ 停止", use_container_width=True):
            sim.stop()
    with col_btn4:
        if st.button("🔄 重置", use_container_width=True):
            sim.stop()
            st.session_state.flight_sim = FlightSimulator(
                st.session_state.planned_waypoints,
                speed_mps=st.session_state.flight_speed,
                battery_reserve_percent=st.session_state.battery_reserve_percent
            )
            sim = st.session_state.flight_sim
    
    sim.update()
    
    status_text = "▶ 飞行中" if sim.is_running else ("⏸ 已暂停" if sim.is_paused else "⏹ 已停止")
    st.markdown(f"<div style='text-align:center; font-size:20px; margin:10px 0;'>{status_text}</div>", 
                unsafe_allow_html=True)
    
    total_wp = len(st.session_state.planned_waypoints)
    current_wp = min(sim.current_index + 1, total_wp)
    elapsed_str = f"{int(sim.elapsed_seconds//60):02d}:{int(sim.elapsed_seconds%60):02d}"
    remaining_dist = max(0, sim.total_distance - sim.dist_traveled)
    remaining_time = remaining_dist / sim.speed if sim.speed > 0 else 0
    eta_str = f"{int(remaining_time//60):02d}:{int(remaining_time%60):02d}" if remaining_time < 3600 else ">1h"
    progress = sim.dist_traveled / sim.total_distance if sim.total_distance > 0 else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("当前航点", f"{current_wp}/{total_wp}")
    col2.metric("飞行速度", f"{sim.speed:.1f} m/s")
    col3.metric("已用时间", elapsed_str)
    col4.metric("剩余距离", f"{remaining_dist/1000:.2f} km")
    col5.metric("预计到达", eta_str)
    
    st.metric("🔋 电量模拟", f"{sim.battery_percent:.0f}%")
    st.progress(progress, text=f"任务进度 {progress*100:.0f}%")

    # ===== 新增：飞行事件日志 =====
    # 记录状态变化事件
    current_status = "飞行中" if sim.is_running else ("已暂停" if sim.is_paused else "已停止")
    last_status = st.session_state.get("last_flight_status", None)
    if current_status != last_status:
        event_time = datetime.datetime.now(BEIJING_TZ).strftime('%H:%M:%S')
        st.session_state.flight_events.append(f"[{event_time}] 状态切换为: {current_status}")
        st.session_state.last_flight_status = current_status
    # 检测到达终点
    if progress >= 1.0 and not st.session_state.get("arrived_logged", False):
        event_time = datetime.datetime.now(BEIJING_TZ).strftime('%H:%M:%S')
        st.session_state.flight_events.append(f"[{event_time}] 🎯 到达终点！")
        st.session_state.arrived_logged = True
    if progress < 1.0:
        st.session_state.arrived_logged = False
    if len(st.session_state.flight_events) > 200:
        st.session_state.flight_events = st.session_state.flight_events[-200:]
    
    with st.expander("📝 飞行事件日志", expanded=False):
        if st.session_state.flight_events:
            for evt in reversed(st.session_state.flight_events[-50:]):
                st.text(evt)
        else:
            st.info("暂无事件，请开始任务")
    st.markdown("---")

    # 地图与拓扑
    map_col, topo_col = st.columns([2, 1])
    with map_col:
        st.subheader("实时飞行地图（自动跟随无人机）")
        center_lat, center_lon = sim.current_pos[1], sim.current_pos[0]
        # ===== 新增：在任务执行页面增加 OSM 底图选择 =====
        use_osm = st.checkbox("使用 OpenStreetMap 底图（推荐）", value=True, key="flight_use_osm")
        if use_osm:
            tiles_url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attr = "OpenStreetMap"
        else:
            if st.session_state.map_style == "卫星影像":
                tiles_url = f"https://webst01.is.autonavi.com/appmaptile?style=6&x={{x}}&y={{y}}&z={{z}}&key={AMAP_KEY}"
                attr = "高德卫星图"
            else:
                tiles_url = f"https://webrd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={{x}}&y={{y}}&z={{z}}&key={AMAP_KEY}"
                attr = "高德矢量街道图"
        m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles=tiles_url, attr=attr)
        
        line_points = [[p[1], p[0]] for p in st.session_state.planned_waypoints]
        folium.PolyLine(line_points, color="blue", weight=3, opacity=0.6, tooltip="规划航线").add_to(m)
        
        if sim.dist_traveled > 0:
            flown = []
            dist_acc = 0
            waypts = st.session_state.planned_waypoints
            for i in range(len(waypts)-1):
                seg_dist = haversine(waypts[i][0], waypts[i][1], waypts[i+1][0], waypts[i+1][1]) * 1000
                if dist_acc + seg_dist < sim.dist_traveled - 1e-6:
                    flown.append(waypts[i+1])
                    dist_acc += seg_dist
                else:
                    t = (sim.dist_traveled - dist_acc) / seg_dist if seg_dist > 0 else 0
                    lon = waypts[i][0] + t * (waypts[i+1][0] - waypts[i][0])
                    lat = waypts[i][1] + t * (waypts[i+1][1] - waypts[i][1])
                    flown.append((lon, lat))
                    break
            if flown:
                flown_path = [[p[1], p[0]] for p in flown]
                folium.PolyLine(flown_path, color="green", weight=5, opacity=0.9, tooltip="已飞路径").add_to(m)
        
        folium.Marker(location=[sim.current_pos[1], sim.current_pos[0]],
                      icon=folium.Icon(color='red', icon='plane', prefix='fa'), 
                      popup="当前位置").add_to(m)
        if st.session_state.planned_waypoints:
            start = st.session_state.planned_waypoints[0]
            end = st.session_state.planned_waypoints[-1]
            folium.Marker(location=[start[1], start[0]], icon=folium.Icon(color='green', icon='play', prefix='fa'), popup="起点").add_to(m)
            folium.Marker(location=[end[1], end[0]], icon=folium.Icon(color='red', icon='stop', prefix='fa'), popup="终点").add_to(m)
        
        if st.checkbox("显示障碍物", value=True, key="flight_show_obs"):
            for obs in st.session_state.obstacles:
                coords = [[lat, lng] for lng, lat in obs['coordinates']]
                height = obs.get('height', 50)
                color = 'darkred' if height >= st.session_state.flight_altitude else 'red'
                folium.Polygon(locations=coords, color=color, weight=2, fill=True, fill_opacity=0.2, 
                               popup=f"{obs['name']} ({height}m)").add_to(m)
        st_folium(m, width=700, height=500, key="flight_map")
    
    with topo_col:
        st.subheader("📡 通信链路拓扑与数据流")
        heart_stats = st.session_state.simulator.get_statistics()
        delay = heart_stats['avg_delay']
        loss = heart_stats['packet_loss_rate']
        st.markdown(link_topology_html(delay, loss), unsafe_allow_html=True)
        st.caption("数据来自实时心跳模拟")
    
    if sim.is_running:
        time.sleep(0.5)
        st.rerun()

# ==================== 页面3：航线规划 ====================
elif st.session_state.page == "航线规划":
    st.header("🗺️ 航线规划 · 多路径选择 + 垂直悬停点")
    
    left_col, right_col = st.columns([1, 2])
    with left_col:
        st.subheader("📍 标记点管理")
        with st.form("add_point_form"):
            lat = st.number_input("纬度", value=39.9042, format="%.6f", key="point_lat")
            lon = st.number_input("经度", value=116.4074, format="%.6f", key="point_lon")
            name = st.text_input("名称", placeholder="例如：测试点")
            submitted = st.form_submit_button("➕ 添加标记点")
            if submitted:
                if st.session_state.input_coordinate_system == "WGS-84":
                    lng_gcj, lat_gcj = wgs84_to_gcj02(lon, lat)
                else:
                    lng_gcj, lat_gcj = lon, lat
                st.session_state.map_points.append({
                    "name": name if name else f"点{len(st.session_state.map_points)+1}",
                    "lat_gcj": lat_gcj,
                    "lon_gcj": lng_gcj,
                    "original_lat": lat,
                    "original_lon": lon,
                    "original_crs": st.session_state.input_coordinate_system
                })
                st.success(f"已添加 {name}")
        if st.button("🗑️ 清空所有标记点", use_container_width=True):
            st.session_state.map_points = []
        
        st.divider()
        st.subheader("✈️ 航线起终点 (A/B点)")
        st.caption(f"当前输入坐标系: **{st.session_state.input_coordinate_system}**")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            a_lat = st.number_input("A点纬度", value=32.2322, format="%.6f", key="a_lat")
            a_lon = st.number_input("A点经度", value=118.7490, format="%.6f", key="a_lon")
            if st.button("设置 A点", use_container_width=True):
                if st.session_state.input_coordinate_system == "WGS-84":
                    lng_gcj, lat_gcj = wgs84_to_gcj02(a_lon, a_lat)
                else:
                    lng_gcj, lat_gcj = a_lon, a_lat
                st.session_state.a_point = {
                    "lat_gcj": lat_gcj,
                    "lon_gcj": lng_gcj,
                    "original_lat": a_lat,
                    "original_lon": a_lon,
                    "original_crs": st.session_state.input_coordinate_system,
                    "name": "A点"
                }
                st.success(f"A点已设")
        with col_a2:
            b_lat = st.number_input("B点纬度", value=32.2343, format="%.6f", key="b_lat")
            b_lon = st.number_input("B点经度", value=118.7490, format="%.6f", key="b_lon")
            if st.button("设置 B点", use_container_width=True):
                if st.session_state.input_coordinate_system == "WGS-84":
                    lng_gcj, lat_gcj = wgs84_to_gcj02(b_lon, b_lat)
                else:
                    lng_gcj, lat_gcj = b_lon, b_lat
                st.session_state.b_point = {
                    "lat_gcj": lat_gcj,
                    "lon_gcj": lng_gcj,
                    "original_lat": b_lat,
                    "original_lon": b_lon,
                    "original_crs": st.session_state.input_coordinate_system,
                    "name": "B点"
                }
                st.success(f"B点已设")
        if st.button("清除 A/B 点", use_container_width=True):
            st.session_state.a_point = None
            st.session_state.b_point = None
            st.success("已清除航线起终点")
        
        st.divider()
        st.subheader("🚁 飞行参数设置")
        altitude = st.slider("巡航高度 (米)", min_value=0, max_value=1000, value=int(st.session_state.flight_altitude), step=10)
        st.session_state.flight_altitude = float(altitude)
        
        st.divider()
        st.subheader("🔄 智能避障设置")
        avoidance_enabled = st.checkbox("启用智能避障", value=st.session_state.avoidance_enabled)
        st.session_state.avoidance_enabled = avoidance_enabled
        
        if avoidance_enabled:
            safe_distance_m = st.slider(
                "绕行安全距离 (米)", 
                min_value=10, 
                max_value=500, 
                value=int(st.session_state.safe_distance * 1000),
                step=10
            )
            st.session_state.safe_distance = safe_distance_m / 1000.0
            
            route_side = st.radio(
                "绕行侧选择",
                ["最优路径", "左侧绕行", "右侧绕行"],
                index=["最优路径", "左侧绕行", "右侧绕行"].index(st.session_state.route_side)
            )
            st.session_state.route_side = route_side
            
            curve_smooth = st.checkbox("显示平滑曲线路径", value=st.session_state.curve_smooth)
            st.session_state.curve_smooth = curve_smooth
        
        st.divider()
        st.subheader("🛬 降落安全设置")
        landing_safety = st.checkbox("启用垂直悬停点（距终点10米，垂直于航向并避开障碍物）", value=st.session_state.landing_safety)
        st.session_state.landing_safety = landing_safety
        if landing_safety:
            st.caption("若终点10米内有障碍物，无人机将悬停于航线垂直方向的10米外安全点（且满足全局安全距离）。")
        
        st.divider()
        st.subheader("🗺️ 地图底图样式")
        style_choice = st.radio("选择地图类型", ["卫星影像", "矢量街道"], index=0 if st.session_state.map_style == "卫星影像" else 1)
        st.session_state.map_style = style_choice
        use_osm = st.checkbox("使用 OpenStreetMap 底图", value=False)
        
        st.divider()
        st.subheader("⚠️ 碰撞检测与航线规划")
        show_obstacles = st.checkbox("在地图上显示障碍物区域", value=True)
        
        if st.session_state.a_point and st.session_state.b_point:
            start_point = (st.session_state.a_point['lon_gcj'], st.session_state.a_point['lat_gcj'])
            end_point = (st.session_state.b_point['lon_gcj'], st.session_state.b_point['lat_gcj'])
            
            blocking = []
            for obs in st.session_state.obstacles:
                if obs.get('height', 50) >= st.session_state.flight_altitude:
                    poly = obs['coordinates']
                    if line_polygon_intersect(start_point, end_point, poly):
                        blocking.append(obs)
            
            original_dist = haversine(start_point[0], start_point[1], end_point[0], end_point[1])
            segments = []
            total_dist = original_dist
            if avoidance_enabled and blocking:
                side_key = {"最优路径": "optimal", "左侧绕行": "left", "右侧绕行": "right"}[st.session_state.route_side]
                segments, total_dist = find_path_with_side(
                    start_point, end_point,
                    st.session_state.obstacles,
                    st.session_state.flight_altitude,
                    st.session_state.safe_distance,
                    side_key
                )
            else:
                segments = [(start_point, end_point)]
            
            waypoints = []
            for seg in segments:
                if not waypoints:
                    waypoints.append(seg[0])
                waypoints.append(seg[1])
            st.session_state.planned_waypoints = waypoints
            
            hover_point = None
            if st.session_state.landing_safety and st.session_state.b_point:
                safe, dist_to_obs, obs_name = check_landing_safety(end_point, st.session_state.obstacles, st.session_state.flight_altitude, safe_radius_km=st.session_state.safe_distance)
                if not safe:
                    last_seg = segments[-1]
                    seg_start = last_seg[0]
                    seg_end = last_seg[1]
                    dx = seg_end[0] - seg_start[0]
                    dy = seg_end[1] - seg_start[1]
                    length = sqrt(dx*dx + dy*dy)
                    if length > 1e-9:
                        ux = dx / length
                        uy = dy / length
                        hover_point, is_perp = get_safe_hover_point(seg_end, (ux, uy), 
                                                                    st.session_state.obstacles, 
                                                                    st.session_state.flight_altitude, 
                                                                    distance_m=10.0,
                                                                    safe_dist_km=st.session_state.safe_distance)
                        if hover_point:
                            st.session_state.planned_waypoints = st.session_state.planned_waypoints[:-1] + [hover_point]
                            st.markdown(f'<div class="warning-text-yellow">⚠️ 降落点距离障碍物“{obs_name}”仅 {dist_to_obs*1000:.1f} 米，已在垂直方向10米外设置安全悬停点（满足安全距离{st.session_state.safe_distance*1000:.0f}m）。</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="warning-text-yellow">⚠️ 无法找到垂直安全点，已保持原航线。</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="safe-text">✅ 降落点安全，距离最近障碍物 {dist_to_obs*1000:.1f} 米（安全距离要求{st.session_state.safe_distance*1000:.0f}m）。</div>', unsafe_allow_html=True)
            
            if avoidance_enabled and blocking:
                extra = total_dist - original_dist
                st.markdown(f'<div class="info-text">✨ {st.session_state.route_side} | 总距离 {total_dist:.3f} km (+{extra:.3f} km) | 安全距离 {st.session_state.safe_distance*1000:.0f}m</div>', unsafe_allow_html=True)
            elif blocking:
                st.markdown(f'<div class="danger-text">⚠️ 危险：航线与 {len(blocking)} 个障碍物相交！请启用智能避障</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="safe-text">✅ 安全：直线距离 {original_dist:.3f} km，所有障碍物均在安全距离外。</div>', unsafe_allow_html=True)
        else:
            st.info("请先设置 A 点和 B 点")
    
    with right_col:
        if AMAP_KEY == "你的高德Key" and not use_osm:
            st.error("⚠️ 请填写高德 Key 或使用 OSM 底图")
        else:
            if use_osm:
                tiles_url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attr = "OpenStreetMap"
            else:
                if st.session_state.map_style == "卫星影像":
                    tiles_url = f"https://webst01.is.autonavi.com/appmaptile?style=6&x={{x}}&y={{y}}&z={{z}}&key={AMAP_KEY}"
                    attr = "高德卫星图"
                else:
                    tiles_url = f"https://webrd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={{x}}&y={{y}}&z={{z}}&key={AMAP_KEY}"
                    attr = "高德矢量街道图"
            
            center_lat, center_lon = 32.2332, 118.7490
            if st.session_state.a_point:
                center_lat, center_lon = st.session_state.a_point['lat_gcj'], st.session_state.a_point['lon_gcj']
            elif st.session_state.b_point:
                center_lat, center_lon = st.session_state.b_point['lat_gcj'], st.session_state.b_point['lon_gcj']
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles=tiles_url, attr=attr)
            
            for point in st.session_state.map_points:
                folium.Marker(location=[point['lat_gcj'], point['lon_gcj']], popup=point['name'], icon=folium.Icon(color='blue')).add_to(m)
            if st.session_state.a_point:
                folium.Marker(location=[st.session_state.a_point['lat_gcj'], st.session_state.a_point['lon_gcj']], popup="A点", icon=folium.Icon(color='green', icon='play', prefix='fa')).add_to(m)
            if st.session_state.b_point:
                folium.Marker(location=[st.session_state.b_point['lat_gcj'], st.session_state.b_point['lon_gcj']], popup="B点", icon=folium.Icon(color='red', icon='stop', prefix='fa')).add_to(m)
            
            if st.session_state.a_point and st.session_state.b_point:
                start_pt = (st.session_state.a_point['lon_gcj'], st.session_state.a_point['lat_gcj'])
                end_pt = (st.session_state.b_point['lon_gcj'], st.session_state.b_point['lat_gcj'])
                
                need_avoid = False
                if st.session_state.avoidance_enabled:
                    for obs in st.session_state.obstacles:
                        if obs.get('height', 50) >= st.session_state.flight_altitude:
                            if line_polygon_intersect(start_pt, end_pt, obs['coordinates']):
                                need_avoid = True
                                break
                
                final_segments = []
                if need_avoid and st.session_state.avoidance_enabled:
                    side_key = {"最优路径": "optimal", "左侧绕行": "left", "右侧绕行": "right"}[st.session_state.route_side]
                    final_segments, _ = find_path_with_side(start_pt, end_pt, st.session_state.obstacles, st.session_state.flight_altitude, st.session_state.safe_distance, side_key)
                else:
                    final_segments = [(start_pt, end_pt)]
                
                display_hover = None
                if st.session_state.landing_safety:
                    safe, _, _ = check_landing_safety(end_pt, st.session_state.obstacles, st.session_state.flight_altitude, st.session_state.safe_distance)
                    if not safe and final_segments:
                        last_seg = final_segments[-1]
                        seg_start = last_seg[0]
                        seg_end = last_seg[1]
                        dx = seg_end[0] - seg_start[0]
                        dy = seg_end[1] - seg_start[1]
                        length = sqrt(dx*dx + dy*dy)
                        if length > 1e-9:
                            ux = dx / length
                            uy = dy / length
                            display_hover, _ = get_safe_hover_point(seg_end, (ux, uy), 
                                                                    st.session_state.obstacles, 
                                                                    st.session_state.flight_altitude,
                                                                    distance_m=10.0,
                                                                    safe_dist_km=st.session_state.safe_distance)
                            if display_hover:
                                final_segments = final_segments[:-1] + [(seg_start, display_hover)]
                
                colors = ['#00FF00', '#00BFFF', '#1E90FF', '#32CD32']
                polyline_pts = [final_segments[0][0]]
                for seg in final_segments:
                    polyline_pts.append(seg[1])
                    line_pts = [[seg[0][1], seg[0][0]], [seg[1][1], seg[1][0]]]
                    folium.PolyLine(line_pts, color=colors[len(polyline_pts)%len(colors)], weight=4, opacity=0.8).add_to(m)
                for i in range(1, len(polyline_pts)-1):
                    wp = polyline_pts[i]
                    folium.CircleMarker(location=[wp[1], wp[0]], radius=6, color='orange', fill=True, popup=f"绕行点 {i}").add_to(m)
                if display_hover:
                    folium.CircleMarker(location=[display_hover[1], display_hover[0]], radius=10, color='purple', fill=True, popup="悬停点 (垂直10m)").add_to(m)
                if st.session_state.curve_smooth and len(polyline_pts) >= 2:
                    try:
                        smooth_pts = bezier_curve(polyline_pts, num_points=100)
                        smooth_line = [[p[1], p[0]] for p in smooth_pts]
                        folium.PolyLine(smooth_line, color='#FF69B4', weight=3, opacity=0.7, dash_array='5,5').add_to(m)
                    except:
                        pass
                total_dist = sum(haversine(s[0][0], s[0][1], s[1][0], s[1][1]) for s in final_segments)
                original_dist = haversine(start_pt[0], start_pt[1], end_pt[0], end_pt[1])
                extra = total_dist - original_dist
                folium.map.Marker(
                    [(start_pt[1]+end_pt[1])/2, (start_pt[0]+end_pt[0])/2],
                    icon=folium.DivIcon(html=f'<div style="font-size:11px; background:rgba(0,0,0,0.7); color:white; padding:2px 6px; border-radius:12px;">✈️ {total_dist:.2f}km (+{extra:.2f})</div>')
                ).add_to(m)
            
            if show_obstacles:
                for obs in st.session_state.obstacles:
                    coords = [[lat, lng] for lng, lat in obs['coordinates']]
                    height = obs.get('height', 50)
                    color = 'darkred' if height >= st.session_state.flight_altitude else 'red'
                    folium.Polygon(locations=coords, color=color, weight=3, fill=True, fill_opacity=0.3, popup=f"{obs['name']}<br>高度: {height}m").add_to(m)
            
            draw = Draw(draw_options={'polygon': {'allowIntersection': False, 'showArea': True}}, edit_options={'edit': True, 'remove': True})
            draw.add_to(m)
            output = st_folium(m, width=700, height=500, key="planning_map")
            
            if output and 'last_active_drawing' in output and output['last_active_drawing']:
                drawing = output['last_active_drawing']
                if drawing and drawing.get('geometry', {}).get('type') == 'Polygon':
                    coords = drawing['geometry']['coordinates'][0]
                    coords = [[c[0], c[1]] for c in coords]
                    new_name = f"障碍物_{len(st.session_state.obstacles)+1}"
                    st.session_state.obstacles.append({"id": str(int(time.time()*1000)), "name": new_name, "coordinates": coords, "height": 50.0})
                    save_obstacles(st.session_state.obstacles)
                    st.success(f"已添加: {new_name}")
                    st.rerun()

# ==================== 页面4：障碍物管理 ====================
elif st.session_state.page == "障碍物管理":
    st.header("⛔ 障碍物管理")
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.subheader("📋 障碍物列表")
        if not st.session_state.obstacles:
            st.info("暂无障碍物")
        else:
            for idx, obs in enumerate(st.session_state.obstacles):
                with st.expander(f"📐 {obs['name']}"):
                    st.write(f"顶点数: {len(obs['coordinates'])}")
                    new_height = st.number_input("高度 (米)", value=int(obs.get('height',50)), step=10, key=f"h_{idx}")
                    if new_height != obs.get('height',50):
                        obs['height'] = float(new_height)
                        save_obstacles(st.session_state.obstacles)
                    new_name = st.text_input("名称", value=obs['name'], key=f"n_{idx}")
                    if new_name != obs['name']:
                        obs['name'] = new_name
                        save_obstacles(st.session_state.obstacles)
                    if st.button("删除", key=f"d_{idx}"):
                        del st.session_state.obstacles[idx]
                        save_obstacles(st.session_state.obstacles)
                        st.rerun()
        if st.button("清空所有", use_container_width=True):
            st.session_state.obstacles = []
            save_obstacles([])
            st.rerun()
        st.divider()
        st.subheader("导入/导出")
        uploaded = st.file_uploader("导入 JSON", type=["json"])
        if uploaded:
            try:
                data = json.load(uploaded)
                if isinstance(data, list):
                    st.session_state.obstacles = data
                    save_obstacles(data)
                    st.success("导入成功")
                    st.rerun()
            except: st.error("无效文件")
        if st.button("导出 JSON"):
            json_str = json.dumps(st.session_state.obstacles, ensure_ascii=False, indent=2)
            st.download_button("下载", data=json_str, file_name="obstacles.json")
    with col_right:
        st.info("""
        📌 **障碍物管理说明**
        - 支持导入/导出 JSON 格式。
        - 高度设置影响避障和降落安全检测。
        - 现在规划的路径会强制保持用户设定的安全距离。
        """)

# ==================== 页面5：坐标系设置 ====================
elif st.session_state.page == "坐标系设置":
    st.header("🌐 坐标系设置")
    crs = st.radio("输入坐标系", ["WGS-84", "GCJ-02"], 
                   index=0 if st.session_state.input_coordinate_system == "WGS-84" else 1)
    st.session_state.input_coordinate_system = "WGS-84" if crs == "WGS-84" else "GCJ-02"
    st.success(f"当前: {st.session_state.input_coordinate_system}")
    st.divider()
    st.subheader("坐标转换测试")
    test_lon = st.number_input("经度", value=118.7490, format="%.6f")
    test_lat = st.number_input("纬度", value=32.2332, format="%.6f")
    if st.button("WGS-84 → GCJ-02"):
        gcj_lon, gcj_lat = wgs84_to_gcj02(test_lon, test_lat)
        st.write(f"GCJ-02: {gcj_lat:.6f}, {gcj_lon:.6f}")

# ==================== 自动刷新（心跳） ====================
if st.session_state.running:
    time.sleep(refresh_rate)
    st.rerun()
