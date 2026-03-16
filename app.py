import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time, sys

# MediaPipe 
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]
LANDMARK_NAMES = [
    "WRIST","TH_CMC","TH_MCP","TH_IP","TH_TIP",
    "IX_MCP","IX_PIP","IX_DIP","IX_TIP",
    "MD_MCP","MD_PIP","MD_DIP","MD_TIP",
    "RG_MCP","RG_PIP","RG_DIP","RG_TIP",
    "PK_MCP","PK_PIP","PK_DIP","PK_TIP",
]

import os

# Get absolute path for the model to prevent errors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

# Liquid Material Modes (The "Alchemy" Update)
MODES = [
    {
        "name": "Deep Water",
        "grav": 35.0, "damp": 0.985, "N": 380,
        "col_low": (180, 30, 10), "col_high": (255, 210, 100),
        "desc": "Classic blue fluid"
    },
    {
        "name": "Liquid Mercury",
        "grav": 60.0, "damp": 0.965, "N": 300,
        "col_low": (140, 140, 140), "col_high": (245, 245, 245),
        "desc": "Heavy, dense silver"
    },
    {
        "name": "Toxic Sludge",
        "grav": 22.0, "damp": 0.992, "N": 350,
        "col_low": (10, 160, 20), "col_high": (140, 255, 100),
        "desc": "Slow, viscous green"
    },
    {
        "name": "Molten Gold",
        "grav": 40.0, "damp": 0.982, "N": 320,
        "col_low": (0, 80, 160), "col_high": (80, 220, 255),
        "desc": "Valuable flowing metal"
    }
]
curr_mode_idx = 0

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=base_options, 
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6))

# Camera 
def open_camera():
    for idx in range(5):
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened(): cap.release(); continue
            time.sleep(1.5)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            for _ in range(20): cap.read()
            ok, frm = cap.read()
            if ok and frm is not None and frm.size > 0:
                print(f"✓ Camera index={idx}")
                return cap
            cap.release()
    return None

print("Finding camera...")
cap = open_camera()
if cap is None:
    print("No camera found"); sys.exit(1)

W_FRAME, H_FRAME = 640, 480
CX, CY = 430, 160
FOCAL  = 20

def project(pts3d):
    return [(int(CX + x * FOCAL), int(CY - y * FOCAL)) for x, y, z in pts3d]

# Rotation
def Rx(a):
    a = np.radians(a); c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def Ry(a):
    a = np.radians(a); c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rz(a):
    a = np.radians(a); c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def R_mat(ax, ay, az):
    return Rz(az) @ Ry(ay) @ Rx(ax)

# Cube
CORNERS = np.array([
    [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
    [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1],
], dtype=float)
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]
FACES = [
    ([4,5,6,7], np.array([0., 0., 1.]),  (10,10,70)),
    ([0,3,2,1], np.array([0., 0.,-1.]),  (10,10,50)),
    ([1,2,6,5], np.array([1., 0., 0.]),  (10,50,80)),
    ([0,4,7,3], np.array([-1.,0., 0.]),  (10,50,80)),
    ([3,7,6,2], np.array([0., 1., 0.]),  (10,70,110)),
    ([0,1,5,4], np.array([0.,-1., 0.]),  (10,70,110)),
]

def draw_cube(frame, R, half_edge):
    pts3d = (CORNERS * half_edge) @ R.T
    p2    = project(pts3d)
    face_order = sorted(range(len(FACES)),
                        key=lambda i: np.mean(pts3d[FACES[i][0]], axis=0)[2])
    overlay = frame.copy()
    for fi in face_order:
        verts, local_n, col = FACES[fi]
        if (R @ local_n)[2] < 0: continue
        cv2.fillPoly(overlay,
                     [np.array([p2[v] for v in verts], dtype=np.int32)], col)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    for s, e in EDGES:
        cv2.line(frame, p2[s], p2[e], (0,0,0), 2, cv2.LINE_AA)
    return pts3d, p2

#  Liquid physics  –  vectorised PBD

N_MAX       = 500
REPULSE_R   = 0.22
REPULSE_K   = 0.058
WALL_MARGIN = 0.10
WALL_K      = 15.0
POKE_FORCE  = 12.0
POKE_R      = 60.0 # Screen pixels
BOUND       = 1.0
DT          = 0.04
VEL_CAP     = 4.0



class Liquid:
    def __init__(self):
        self.pos      = np.random.uniform(-BOUND*0.7, BOUND*0.7, (N_MAX, 3))
        self.pos[:,1] = np.random.uniform(-BOUND*0.8, -0.05, N_MAX)
        self.vel      = np.zeros((N_MAX, 3))

    def _wall_force(self):
        f = np.zeros_like(self.pos)
        for axis in range(3):
            d_pos = BOUND - self.pos[:, axis]
            d_neg = BOUND + self.pos[:, axis]
            cp = d_pos < WALL_MARGIN
            cn = d_neg < WALL_MARGIN
            if cp.any():
                f[cp, axis] -= WALL_K * (1.0 - d_pos[cp] / WALL_MARGIN) ** 2
            if cn.any():
                f[cn, axis] += WALL_K * (1.0 - d_neg[cn] / WALL_MARGIN) ** 2
        return f

    def _repulse(self, pos):
        diff  = pos[:, None, :] - pos[None, :, :]
        dist2 = (diff ** 2).sum(axis=2)
        np.fill_diagonal(dist2, 1e9)
        mask  = dist2 < (REPULSE_R ** 2)
        if not mask.any():
            return pos
        dist  = np.sqrt(np.where(mask, dist2, 1.0))
        over  = np.where(mask, REPULSE_R - dist, 0.0)
        safe  = np.where(mask[:,:,None], diff / dist[:,:,None], 0.0)
        delta = (safe * (over * REPULSE_K * 0.5)[:,:,None]).sum(axis=1)
        return pos + delta

    def update(self, ax, ay, az, poke_pos=None):
        m = MODES[curr_mode_idx]
        N = m['N']
        R = R_mat(ax, ay, az)
        g_local = R.T @ np.array([0.0, -m['grav'], 0.0])

        self.vel[:N] += (g_local + self._wall_force()[:N]) * DT
        
        # Fingertip Poke Interaction
        if poke_pos is not None:
            pts_proj = project((self.pos[:N] * half_edge) @ R.T)
            for i in range(N):
                dx = pts_proj[i][0] - poke_pos[0]
                dy = pts_proj[i][1] - poke_pos[1]
                dist2 = dx*dx + dy*dy
                if dist2 < POKE_R**2:
                    dist = np.sqrt(dist2) + 1e-6
                    # Push away in 3D based on 2D proximity
                    push = (np.array([dx, -dy, 0.0]) / dist) * POKE_FORCE
                    self.vel[i] += push * R.T @ np.array([1, 1, 1]) * 0.5

        pred = self.pos[:N] + self.vel[:N] * DT
        pred = self._repulse(pred)

        self.vel[:N] = (pred - self.pos[:N]) / DT

        for axis in range(3):
            hi = pred[:, axis] >  BOUND
            lo = pred[:, axis] < -BOUND
            self.vel[:N, axis][hi] = -abs(self.vel[:N, axis][hi]) * 0.25
            self.vel[:N, axis][lo] =  abs(self.vel[:N, axis][lo]) * 0.25
            pred[hi, axis] =  BOUND - 1e-4
            pred[lo, axis] = -BOUND + 1e-4

        self.vel[:N] *= m['damp']
        self.vel[:N] = np.clip(self.vel[:N], -VEL_CAP, VEL_CAP)
        self.pos[:N] = pred

        bad = ~np.isfinite(self.pos[:N]).all(axis=1)
        if bad.any():
            self.pos[:N][bad] = np.random.uniform(-0.3, 0.3, (bad.sum(), 3))
            self.vel[:N][bad] = 0.0

    #  Blue liquid renderer
    # Instead of individual circles, we splat each particle as a
    # soft gaussian blob, then composite with the frame so it looks
    # like a continuous fluid surface rather than discrete balls.
    def draw(self, frame, half_edge, ax, ay, az):
        m = MODES[curr_mode_idx]
        N = m['N']
        R      = R_mat(ax, ay, az)
        pts    = (self.pos[:N] * half_edge) @ R.T
        p2     = project(pts)
        depths = pts[:, 2]
        speeds = np.linalg.norm(self.vel[:N], axis=1)

        d_min, d_max = depths.min(), depths.max() + 1e-5
        s_min, s_max = speeds.min(), speeds.max() + 1e-5

        order = np.argsort(depths)
        glow = np.zeros((H_FRAME, W_FRAME, 3), dtype=np.float32)

        # Mode-based colors
        c_low, c_high = np.array(m['col_low']), np.array(m['col_high'])

        for i in order:
            px, py = p2[i]
            if not (0 < px < W_FRAME and 0 < py < H_FRAME): continue
            d = (depths[i] - d_min) / (d_max - d_min)
            s = (speeds[i] - s_min) / (s_max - s_min)

            blob_r = int(np.clip(9 - d * 4, 4, 11))
            
            # Dynamic color interpolation based on speed/depth
            mix = np.clip(s * 0.7 + (1-d) * 0.3, 0, 1)
            col = c_low * (1 - mix) + c_high * mix
            
            # Reflection Mapping: Sample camera frame color for extra realism
            ref_col = frame[py, px].astype(np.float32)
            col = col * 0.8 + ref_col * 0.2

            # Draw a filled circle into the glow layer
            cv2.circle(glow, (px, py), blob_r, col.tolist(), -1)

        # Gaussian blur turns overlapping circles into a smooth fluid blob
        glow = cv2.GaussianBlur(glow, (15, 15), 6)

        # Clamp and convert
        glow_u8 = np.clip(glow, 0, 255).astype(np.uint8)

        # Layer 2: sharp specular highlights on top of the blurred body
        spec = np.zeros_like(frame)
        for i in order:
            px, py = p2[i]
            if not (0 < px < W_FRAME and 0 < py < H_FRAME): continue
            d = (depths[i] - d_min) / (d_max - d_min)
            s = (speeds[i] - s_min) / (s_max - s_min)
            if d > 0.6: continue     # only near particles get highlights
            brightness = int(np.clip(180 + s * 75, 100, 255))
            cv2.circle(spec, (px, py), 2, (brightness, brightness, brightness), -1)

        # Composite: frame → glow (55% opacity) → specular (70% opacity)
        # Where glow has colour, blend it in; black areas = transparent
        glow_mask  = (glow_u8.sum(axis=2) > 15).astype(np.float32)
        blended    = frame.copy().astype(np.float32)
        glow_f     = glow_u8.astype(np.float32)
        frame_f    = frame.astype(np.float32)

        alpha = glow_mask[:, :, None] * 0.72
        blended = frame_f * (1 - alpha) + glow_f * alpha
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # Add specular on top
        cv2.addWeighted(blended, 1.0, spec, 0.70, 0, frame)


liquid = Liquid()

#Fingertip control 
TIP_IDX     = [8, 12, 16, 20]
TIPS_ALL    = [4, 8, 12, 16, 20]
OPEN_THRESH = 0.28

# Spin settings 
# Velocity-based sensitivity: the faster you move your hand,
# the more DEG_PER_PX scales up → quick flick = fast spin.
DEG_PER_PX_BASE = 0.55    # base sensitivity 
DEG_PER_PX_MAX  = 2.2     # max sensitivity on fast moves
SPEED_SCALE     = 0.08    # how quickly sensitivity ramps with movement speed

DEADZONE_PX = 3.0
INERTIA     = 0.88        # high inertia = spin coasts after a flick


def lm_vec(lm, i):
    return np.array([lm[i].x, lm[i].y, lm[i].z])

def is_hand_open(lm):
    wrist = lm_vec(lm, 0)
    scale = np.linalg.norm(lm_vec(lm, 17) - wrist) + 1e-6
    avg_d = np.mean([np.linalg.norm(lm_vec(lm, t) - wrist) for t in TIPS_ALL])
    return (avg_d / scale) > OPEN_THRESH

def fingertip_centroid(lm):
    xs = [lm[i].x * W_FRAME for i in TIP_IDX]
    ys = [lm[i].y * H_FRAME for i in TIP_IDX]
    return np.array([np.mean(xs), np.mean(ys)])

def draw_hand(frame, lm, hand_open, centroid):
    pts      = [(int(l.x*W_FRAME), int(l.y*H_FRAME)) for l in lm]
    bone_col = (0,220,90)   if hand_open else (60,60,200)
    dot_col  = (60,255,140) if hand_open else (80,80,255)
    for s,e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], bone_col, 2, cv2.LINE_AA)
    for i,(px,py) in enumerate(pts):
        cv2.circle(frame, (px,py), 6, bone_col, 1)
        cv2.circle(frame, (px,py), 4, dot_col, -1)
        cv2.putText(frame, str(i), (px+6,py-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)
    for ti in TIP_IDX:
        px, py = pts[ti]
        cv2.circle(frame, (px,py), 10, (0,255,200), 2, cv2.LINE_AA)
        cv2.circle(frame, (px,py),  3, (0,255,200), -1, cv2.LINE_AA)
    cx, cy = int(centroid[0]), int(centroid[1])
    cv2.drawMarker(frame, (cx,cy), (0,255,180),
                   cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
    ov = frame.copy()
    cv2.rectangle(ov, (0,0),(190,H_FRAME),(8,8,8),-1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    label = "OPEN – move to rotate" if hand_open else "FIST – frozen"
    col   = (0,220,90) if hand_open else (60,60,200)
    cv2.putText(frame, "LANDMARKS", (4,13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    cv2.line(frame,(0,17),(190,17),(0,180,70),1)
    for i,l in enumerate(lm):
        y = 28 + i*21
        cv2.putText(frame, f"{i:02d} {LANDMARK_NAMES[i]}", (3,y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.28,(170,230,170),1,cv2.LINE_AA)
        cv2.putText(frame, f"    {l.x:.2f} {l.y:.2f} {l.z:.2f}", (3,y+10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.25,(110,170,110),1,cv2.LINE_AA)

def get_pinch(lm, w, h):
    dx = (lm[4].x - lm[8].x)*w
    dy = (lm[4].y - lm[8].y)*h
    return (dx*dx + dy*dy)**0.5

def smooth(buf, val, n=5):
    buf.append(val); buf[:] = buf[-n:]
    return float(np.mean(buf))

# State
ax, ay, az    = 15.0, 25.0, 0.0
spin_x = spin_y = 0.0
prev_centroid = None
half_edge     = 6.0
ph            = []
fail_cnt      = 0

WIN_NAME = "Blue Liquid Cube"
cv2.namedWindow(WIN_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Blue liquid | Flick fast = fast spin | Fist = freeze | Pinch = resize | Q = quit\n")

# Main loop
while True:
    ok, frame = cap.read()
    if not ok or frame is None or frame.size == 0:
        fail_cnt += 1
        time.sleep(0.03)
        if fail_cnt > 80:
            cap.release(); cap = open_camera()
            if cap is None: break
            fail_cnt = 0
        continue
    fail_cnt = 0
    frame = cv2.flip(frame, 1)

    try:
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    except Exception as e:
        result = None

    hand_detected = False

    if result and result.hand_landmarks:
        lm            = result.hand_landmarks[0]
        hand_detected = True
        hand_open     = is_hand_open(lm)

        pinch     = smooth(ph, get_pinch(lm, W_FRAME, H_FRAME))
        half_edge = float(np.clip(pinch * 0.035, 3.0, 8.0))
        centroid  = fingertip_centroid(lm)

        if hand_open and prev_centroid is not None:
            dx = centroid[0] - prev_centroid[0]
            dy = centroid[1] - prev_centroid[1]
            if abs(dx) < DEADZONE_PX: dx = 0.0
            if abs(dy) < DEADZONE_PX: dy = 0.0

            # Dynamic sensitivity: scale DEG_PER_PX with movement speed
            raw_speed = (dx**2 + dy**2) ** 0.5
            dpp = np.clip(
                DEG_PER_PX_BASE + raw_speed * SPEED_SCALE,
                DEG_PER_PX_BASE,
                DEG_PER_PX_MAX
            )

            target_sy = dx * dpp
            target_sx = dy * dpp

            spin_y = spin_y * INERTIA + target_sy * (1.0 - INERTIA)
            spin_x = spin_x * INERTIA + target_sx * (1.0 - INERTIA)

        elif not hand_open:
            # Fist: coast then stop
            spin_x       *= 0.88
            spin_y       *= 0.88
            prev_centroid = None

        ay = (ay + spin_y) % 360
        ax = (ax + spin_x) % 360
        az = ay * 0.08

        if hand_open:
            prev_centroid = centroid
        draw_hand(frame, lm, hand_open, centroid)
    else:
        spin_x *= 0.94
        spin_y *= 0.94
        ay = (ay + spin_y) % 360
        ax = (ax + spin_x) % 360
        prev_centroid = None

    poke_pos = centroid if (hand_detected and hand_open) else None
    liquid.update(ax, ay, az, poke_pos=poke_pos)
    R = R_mat(ax, ay, az)
    draw_cube(frame, R, half_edge)
    liquid.draw(frame, half_edge, ax, ay, az)

    # Info UI
    hx = 200
    m = MODES[curr_mode_idx]
    status = "HAND DETECTED" if hand_detected else "no hand – coasting"
    cv2.putText(frame, f"[{status}]  Y:{spin_y:+.1f}°  X:{spin_x:+.1f}°",
                (hx, H_FRAME-45), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1, cv2.LINE_AA)
    cv2.putText(frame, f"MODE: {m['name']} ({m['desc']})", 
                (hx, H_FRAME-30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, m['col_high'], 1, cv2.LINE_AA)
    cv2.putText(frame, f"Angle Y:{ay:.0f}  X:{ax:.0f}  |  N:{m['N']}  |  [Q] quit  [M] Liquid Mode",
                (hx, H_FRAME-10),  cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140,140,140), 1, cv2.LINE_AA)

    cv2.imshow(WIN_NAME, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('m'):
        curr_mode_idx = (curr_mode_idx + 1) % len(MODES)

cap.release()
cv2.destroyAllWindows()