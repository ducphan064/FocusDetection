import sys, os, time, math
import numpy as np
import cv2
import mediapipe as mp
import platform
# 1) LANDMARK & HẰNG SỐ

LEFT_EYE  = [33,160,158,133,153,144]
RIGHT_EYE = [263,387,385,362,380,373]

LEFT_EYE_CORNERS  = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)

NOSE_TIP = 1; CHIN = 152; LEFT_EAR = 234; RIGHT_EAR = 454; LEFT_EYE_COR = 33; RIGHT_EYE_COR = 263

##khúc này là mống mắt (iris, 5 điểm là mỗi bên 4 điểm về quanh + 1 tâm)
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


# === MOUTH landmarks (FaceMesh)
MOUTH_LEFT   = 61    # khoé trái (outer)
MOUTH_RIGHT  = 291   # khoé phải (outer)
MOUTH_TOP    = 13    # môi trên giữa (inner)
MOUTH_BOTTOM = 14    # môi dưới giữa (inner)



### Vector đặc trưng khuôn mặc 3D "chuẩn" theo cộng đồng MediaPipe cung cấp
MODEL_PTS = np.array([
    [0.0, 0.0, 0.0],      # nose tip
    [0.0, -63.6, -12.5],  # chin
    [-43.3, 32.7, -26.0], # left eye corner
    [43.3, 32.7, -26.0],  # right eye corner
    [-28.9, -28.9, -24.1],# left ear
    [28.9, -28.9, -24.1], # right ear
], dtype=np.float32)

# 2) TIỆN ÍCH HÌNH HỌC
def dist2d(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def eye_aspect_ratio(pts):  # pts: [(x,y)] * 6
    def d(i, j): return dist2d(pts[i], pts[j])
    return (d(1,5) + d(2,4)) / (2.0*d(0,3) + 1e-8)

def iris_center(landmarks, idxs, w, h):
    xs, ys = [], []
    for i in idxs:
        xs.append(landmarks[i].x * w)
        ys.append(landmarks[i].y * h)
    if not xs: return None
    return (float(np.mean(xs)), float(np.mean(ys)))

def eye_center_and_width(landmarks, corners, w, h):
    i, j = corners
    p1 = (landmarks[i].x * w, landmarks[i].y * h)
    p2 = (landmarks[j].x * w, landmarks[j].y * h)
    center = ((p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0)
    width = dist2d(p1, p2)
    return center, width

def compute_gaze_offset(landmarks, w, h):
    # Chuẩn hoá độ lệch tròng mắt theo bề ngang mắt (0..~0.5)
    cL, wL = eye_center_and_width(landmarks, LEFT_EYE_CORNERS, w, h)
    irisL = iris_center(landmarks, LEFT_IRIS, w, h)
    print("------------------------------")
    print("Left eye center", cL)
    print("Left eye width", wL)
    print("Iris Center Left",irisL)
    offL = 0.0
    if irisL and wL > 1:
        vL = (irisL[0] - cL[0], irisL[1] - cL[1])
        offL = math.hypot(vL[0], vL[1]) / wL

    cR, wR = eye_center_and_width(landmarks, RIGHT_EYE_CORNERS, w, h)
    irisR = iris_center(landmarks, RIGHT_IRIS, w, h)
    offR = 0.0
    if irisR and wR > 1:
        vR = (irisR[0] - cR[0], irisR[1] - cR[1])
        offR = math.hypot(vR[0], vR[1]) / wR

    if offL == 0.0 and offR == 0.0: return 0.0
    if offL == 0.0: return offR
    if offR == 0.0: return offL
    return 0.5*(offL + offR)

def solve_head_pose(landmarks, w, h):
    pts2d = np.array([
        [landmarks[NOSE_TIP].x*w,  landmarks[NOSE_TIP].y*h],
        [landmarks[CHIN].x*w,      landmarks[CHIN].y*h],
        [landmarks[LEFT_EYE_COR].x*w,  landmarks[LEFT_EYE_COR].y*h],
        [landmarks[RIGHT_EYE_COR].x*w, landmarks[RIGHT_EYE_COR].y*h],
        [landmarks[LEFT_EAR].x*w,  landmarks[LEFT_EAR].y*h],
        [landmarks[RIGHT_EAR].x*w, landmarks[RIGHT_EAR].y*h],
    ], dtype=np.float32)
    fx = fy = w
    cx, cy = w/2.0, h/2.0
    cam_mtx = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
    dist = np.zeros((4,1), dtype=np.float32)
    okp, rvec, _ = cv2.solvePnP(MODEL_PTS, pts2d, cam_mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not okp: return None, None
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    pitch = math.degrees(math.atan2(-R[2,0], sy))
    yaw   = math.degrees(math.atan2(R[1,0], R[0,0]))
    return yaw, pitch

def get_eye_points(landmarks, idxs, w, h):
    return [(landmarks[i].x*w, landmarks[i].y*h) for i in idxs]

def mouth_aspect_ratio(landmarks, w, h):
    """MAR = (khoảng mở dọc môi) / (chiều ngang miệng)"""
    lx, ly = landmarks[MOUTH_LEFT].x*w,  landmarks[MOUTH_LEFT].y*h
    rx, ry = landmarks[MOUTH_RIGHT].x*w, landmarks[MOUTH_RIGHT].y*h
    tx, ty = landmarks[MOUTH_TOP].x*w,   landmarks[MOUTH_TOP].y*h
    bx, by = landmarks[MOUTH_BOTTOM].x*w,landmarks[MOUTH_BOTTOM].y*h

    horizontal = math.hypot(rx - lx, ry - ly) + 1e-6
    vertical   = math.hypot(bx - tx, by - ty)
    return vertical / horizontal


# 3) SMOOTHING & FSM
class FeatureSmoother:
    def __init__(self, alpha=0.75):
        self.yaw=self.pitch=self.ear=self.gaze=None
        self.a=alpha
    def ema(self, prev, new):
        return self.a*prev + (1-self.a)*new if prev is not None else new
    def update(self, yaw, pitch, ear, gaze_off):
        if yaw   is not None: self.yaw   = self.ema(self.yaw, yaw)
        if pitch is not None: self.pitch = self.ema(self.pitch, pitch)
        if ear   is not None: self.ear   = self.ema(self.ear, ear)
        if gaze_off is not None: self.gaze = self.ema(self.gaze, gaze_off)
        return self.yaw, self.pitch, self.ear, self.gaze

class FocusFSM:
    def __init__(self, th_on=0.88, th_off=0.55, dwell_on=1.2, dwell_off=0.8):
        self.state = "focused"
        self.th_on = th_on
        self.th_off = th_off
        self.dwell_on = dwell_on
        self.dwell_off = dwell_off
        self._cand_since = None
        self._cand_state = None
    def update(self, score, tnow=None):
        t = tnow or time.time()
        target = need = None
        if self.state == "focused":
            if score >= self.th_on:
                target, need = "unfocused", self.dwell_on
        else:
            if score <= self.th_off:
                target, need = "focused", self.dwell_off
        if target is None:
            self._cand_since = None; self._cand_state = None
            return self.state, 0.0
        if self._cand_state != target:
            self._cand_state = target
            self._cand_since = t
        elapsed = t - self._cand_since
        if elapsed >= need:
            self.state = target
            self._cand_since = None; self._cand_state = None
            return self.state, 1.0
        return self.state, min(1.0, elapsed/need)


class YawnTracker:
    """
    Phát hiện 'ngáp' dựa trên MAR:
      - baseline MAR được học online (EMA) khi miệng không mở quá to
      - yawn nếu MAR > baseline*ratio_on liên tục >= dwell_on
      - reset về 'no_yawn' khi MAR < baseline*ratio_off liên tục >= dwell_off
    """
    def __init__(self, ema_alpha=0.9, ratio_on=2.0, ratio_off=1.4,
                 dwell_on=0.6, dwell_off=0.25, cap_update=1.6):
        self.alpha = ema_alpha
        self.r_on  = ratio_on
        self.r_off = ratio_off
        self.dwell_on  = dwell_on
        self.dwell_off = dwell_off
        self.cap_update = cap_update  # chỉ cập nhật baseline khi MAR < baseline*cap_update

        self.baseline = None
        self.state = "no_yawn"
        self._cand_since = None
        self._cand_target = None

    def _ema(self, prev, new):
        return self.alpha*prev + (1-self.alpha)*new if prev is not None else new

    def update(self, mar, tnow=None):
        t = tnow or time.time()
        if mar is None or math.isinf(mar) or math.isnan(mar):
            return {"state": self.state, "baseline": self.baseline, "ratio": 1.0, "progress": 0.0}

        # 1) cập nhật baseline (chỉ khi không mở miệng quá lớn)
        if self.baseline is None:
            self.baseline = mar
        else:
            if mar < self.baseline * self.cap_update:
                self.baseline = self._ema(self.baseline, mar)

        ratio = mar / max(self.baseline, 1e-6)

        # 2) hysteresis + dwell
        target, need = None, None
        if self.state == "no_yawn":
            if ratio >= self.r_on:
                target, need = "yawn", self.dwell_on
        else:  # yawn
            if ratio <= self.r_off:
                target, need = "no_yawn", self.dwell_off

        if target is None:
            self._cand_since = None
            self._cand_target = None
            return {"state": self.state, "baseline": self.baseline, "ratio": ratio, "progress": 0.0}

        if self._cand_target != target:
            self._cand_target = target
            self._cand_since = t

        elapsed = t - self._cand_since
        if elapsed >= need:
            self.state = target
            self._cand_since = None
            self._cand_target = None
            return {"state": self.state, "baseline": self.baseline, "ratio": ratio, "progress": 1.0}

        return {"state": self.state, "baseline": self.baseline, "ratio": ratio,
                "progress": min(1.0, elapsed/need)}



# 4) SCORING 0..1
def ramp(x, start, end):
    if x<=start: return 0.0
    if x>=end:   return 1.0
    return (x-start)/(end-start)
#Giả sử 10 frame
def compute_score(yaw, pitch, ear, gaze_off, yaw0, pitch0, ear0): ## hàm tính score tổng focus _ unfocus
    dyaw = abs((yaw or 0) - yaw0)
    dpit = abs((pitch or 0) - pitch0)
    # Nhạy vừa phải (đổi để test dễ hơn/khó hơn)
    yaw_score  = ramp(dyaw, 15, 35) #góc quay ngang của đầu
    pit_score  = ramp(dpit, 12, 28) #góc quay dọc của đầu
    blink_score = ramp(max(0.0, (ear0 - (ear or ear0))), 0.06, 0.14) #chớp mắt
    gaze_score  = ramp(gaze_off or 0.0, 0.1, 0.22) #hướng mắt
    print("Blink score", blink_score)
    print('Gaze after ramp', gaze_score)
    w_yaw, w_pit, w_blink, w_gaze = 0.25, 0.25, 0.7, 0.7
    score = (w_yaw*yaw_score + w_pit*pit_score + w_blink*blink_score + w_gaze*gaze_score)
    return min(1.5, score)
def open_camera():
    system = platform.system()

    print("[DEBUG] Detected OS:", system)

    # Chọn backend tùy hệ điều hành
    if system == "Darwin":  # macOS
        return cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    elif system == "Windows":
        return cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow (ổn định nhất)
    else:
        # Linux / Android / Raspberry Pi → V4L2
        return cv2.VideoCapture(0, cv2.CAP_V4L2)
def digital_zoom(frame, zoom_factor=1.5):
    h, w, _ = frame.shape

    nh = int(h / zoom_factor)
    nw = int(w / zoom_factor)

    x1 = (w - nw) // 2
    y1 = (h - nh) // 2
    x2 = x1 + nw
    y2 = y1 + nh

    cropped = frame[y1:y2, x1:x2]

    return cv2.resize(cropped, (w, h))

def put_panel(img, sens,x=10, y=185):
        txt = f"th_on={sens['th_on']:.2f} | dwell_on={sens['dwell_on']:.2f}   hotkeys: [ ]  - ="
        cv2.putText(img, txt, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
